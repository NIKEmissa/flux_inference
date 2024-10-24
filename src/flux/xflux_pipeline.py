import os
from PIL import Image
import numpy as np
import math
import torch
import pdb

from einops import rearrange

from src.flux.modules.layers import DoubleStreamBlockLoraProcessor
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack, get_zj_schedule
from src.flux.util import (load_ae, load_clip, load_flow_model, load_t5, load_controlnet,
                           load_flow_model_quintized, Annotator, get_lora_rank, load_checkpoint)


class XFluxPipeline:
    def __init__(self, base_model_path, model_type, device: list = ["cuda:0"], offload: bool = False, only_offload_t5: bool = False):
        self.base_model_path = base_model_path
        self.device = [torch.device(d) for d in device]
        self.offload = offload
        self.only_offload_t5 = only_offload_t5
        self.model_type = model_type

        self.clip = load_clip(base_model_path, self.device[0])
        if only_offload_t5:
            self.t5 = load_t5(base_model_path, "cpu", max_length=512)
        else:
            self.t5 = load_t5(base_model_path, self.device[0], max_length=512)

        ae_ckpt_path = os.path.join(base_model_path, 'ae.safetensors')
        # self.ae = load_ae(ae_ckpt_path, model_type, device="cpu" if offload else self.device[0])
        self.ae = load_ae(ae_ckpt_path, model_type, device="cpu")
        if not offload:
            # FIXME if use encoder
            self.ae.decoder = self.ae.decoder.to(self.device[0])
            
        if "fp8" in model_type:
            dit_ckpt_path = os.path.join(base_model_path+'-fp8', 'flux-dev-fp8.safetensors')
            # self.model = load_flow_model_quintized(dit_ckpt_path, model_type, device="cpu" if offload else self.device[-1])
            if offload and len(self.device) == 1:
                self.model = load_flow_model_quintized(dit_ckpt_path, model_type, device="cpu")
            else:
                self.model = load_flow_model_quintized(dit_ckpt_path, model_type, self.device[-1])
        else:
            dit_ckpt_path = os.path.join(base_model_path, 'flux1-dev.safetensors')
            # self.model = load_flow_model(dit_ckpt_path, model_type, device="cpu" if offload else self.device[-1])
            if offload and len(self.device) == 1:
                self.model = load_flow_model(dit_ckpt_path, model_type, device="cpu")
            else:
                self.model = load_flow_model(dit_ckpt_path, model_type, self.device[-1])
                self.model.single_blocks = self.model.single_blocks.to(self.device[0])
                # self.model = torch.compile(self.model)
        torch.cuda.empty_cache()

        # FIXME, TODO, 后期需要修改
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(self.device[-1])

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)

        self.control_type = control_type
        self.annotator = Annotator()
        self.controlnet_loaded = True

    def __call__(self,
                 prompt: str,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 true_gs = 3,
                 neg_prompt: str = '',
                 timestep_to_start_cfg: int = 0,
                 seed: int = 1000000,
                 callback=None,
                 cb_infos=None,
                 progress_len=100,
                 msg_queue=None,
                 img_seq_len='default'
                 ):
        width = 16 * width // 16
        height = 16 * height // 16
        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height, self.control_type)
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        return self.forward(prompt, width, height, guidance, num_steps, controlnet_image,
         timestep_to_start_cfg=timestep_to_start_cfg, true_gs=true_gs, neg_prompt=neg_prompt, seed=seed, callback=callback, msg_queue=msg_queue, cb_infos=cb_infos, progress_len=progress_len, img_seq_len=img_seq_len)
        
    @torch.no_grad()
    def forward(self, prompt, width, height, guidance, num_steps, controlnet_image=None, timestep_to_start_cfg=0, true_gs=3, neg_prompt="", seed=10000000, callback=None, msg_queue=None, cb_infos=None, progress_len=100, img_seq_len='default'):
        if callback is not None or msg_queue is not None:
            base_progress = cb_infos['progress']
            
        x = get_noise(
            1, height, width, device=self.device[0],
            dtype=torch.bfloat16, seed=seed
        )

        # if img_seq_len == 'default':
        #     timesteps = get_schedule(
        #         num_steps,
        #         (width // 8) * (height // 8) // (16 * 16),
        #         shift=True,
        #     )
        # elif img_seq_len == 'pro':
        #     timesteps = get_schedule(
        #         num_steps,
        #         (width // 8) * (height // 8) // 4,
        #         shift=True,
        #     )
        # else:
        #     timesteps = get_schedule(
        #         num_steps,
        #         img_seq_len,
        #         shift=True,
        #     )
        # print("img_seq_len is ============> " + str(img_seq_len))  

        ## 这里的img_seq_len等价于mu
        timesteps = get_zj_schedule(
                num_steps,
                img_seq_len,
                shift=True,
            )
        print("mu is ============> " + str(img_seq_len))  
        
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.only_offload_t5:
                self.t5 = self.t5.to(self.device[0])
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device[0]), self.clip.to(self.device[0])
                
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            inp_cond = {k:v.to(self.device[-1]) for k,v  in inp_cond.items()}
            neg_inp_cond = {k:v.to(self.device[-1]) for k,v  in neg_inp_cond.items()}
            torch.cuda.empty_cache()
            if callback is not None:
                cb_infos['code'] = 1
                cb_infos['progress'] = base_progress + int(0.05 * progress_len)
                callback(cb_infos)
            if msg_queue is not None:
                cb_infos['code'] = 1
                cb_infos['progress'] = base_progress + int(0.05 * progress_len)
                msg_queue.put(cb_infos)

            if self.only_offload_t5:
                self.offload_model_to_cpu(self.t5)
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device[-1])
                self.model.single_blocks = self.model.single_blocks.to(self.device[0])
                
            if self.controlnet_loaded:
                x = denoise_controlnet(
                    self.model, **inp_cond, controlnet=self.controlnet,
                    timesteps=timesteps, guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs
                )
            else:
                x = denoise(self.model, **inp_cond, timesteps=timesteps, guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    callback=callback,
                    msg_queue=msg_queue,
                    cb_infos=cb_infos,
                    progress_len=progress_len*0.9
                )

            if self.offload:
                if len(self.device) == 1:
                    self.offload_model_to_cpu(self.model)

            x = unpack(x.float(), height, width)
            x = x.to(self.device[0])
            torch.cuda.empty_cache()
            
            if self.offload:
                self.ae.decoder = self.ae.decoder.to(self.device[0])
                
            x = self.ae.decode(x)
            
            if self.offload:
                self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        if callback is not None:
            cb_infos['code'] = 1
            cb_infos['progress'] = base_progress + int(1.0 * progress_len)
            callback(cb_infos)
        if msg_queue is not None:
            cb_infos['code'] = 1
            cb_infos['progress'] = base_progress + int(1.0 * progress_len)
            msg_queue.put(cb_infos)
        return output_img

    def offload_model_to_cpu(self, *models):
        # if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()

class XFluxImg2ImgPipeline:
    def __init__(self, model_type, device, offload: bool = False, seed: int = None):
        self.device = torch.device(device)
        self.offload = offload
        self.seed = seed
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device,hf_download=False)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(self.device)

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)

        self.control_type = control_type
        self.annotator = Annotator()
        self.controlnet_loaded = True
    def img_preprocess(self,img, width, height):
        img = img.resize((width, height))
        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        img = img.permute(2, 0, 1)
        return img.unsqueeze(0)
    def get_timesteps(self, num_inference_steps, strength, device,timesteps):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        
        t_start_timesteps = timesteps[t_start:]

        return t_start_timesteps, num_inference_steps - t_start
    
    #准备图生图的latent
    def prepare_latents(self, image, timesteps, device, seed,height, width):

        noise = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        image = image.to(device=noise.device, dtype=noise.dtype)

        
        sigma = timesteps[0]

        init_latent = (1 - sigma) * image + sigma * noise
        init_latent = init_latent.type(torch.bfloat16)
        return init_latent
        

    def __call__(self,
                 image,
                 prompt: str = '',
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 true_gs = 3,
                 neg_prompt: str = '',
                 timestep_to_start_cfg: int = 0,
                 timesteps=None,
                 strength=1,
                 latents=None
                 ):
    
        width = 16 * width // 16
        height = 16 * height // 16
        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height, self.control_type)
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        return self.forward(image,prompt, width, height, guidance, num_steps, controlnet_image,
         timestep_to_start_cfg=timestep_to_start_cfg, true_gs=true_gs, neg_prompt=neg_prompt,timesteps=timesteps,strength=strength,latents=latents)

    def forward(self,image, prompt, width, height, guidance, num_steps, controlnet_image=None, timestep_to_start_cfg=0, true_gs=3, neg_prompt="",timesteps=None,strength=1,latents=None):

        #加载图像
    
        image = self.img_preprocess(image, width, height)
        #获取image encoder
        with torch.no_grad():
            image = self.ae.encode(image.to(self.device).to(torch.float32))
     
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
            
        )

        #处理图生图中噪声强度
        timesteps, num_inference_steps = self.get_timesteps(num_steps, strength, self.device,timesteps)

        #获取图生图的latent
        if latents is None:
            init_latent = self.prepare_latents(
                            image,
                            timesteps,
                            self.device,
                            self.seed,
                            height,
                            width
                        )
        else:
            init_latent = latent


        x = init_latent

        torch.manual_seed(self.seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            if self.controlnet_loaded:
                x = denoise_controlnet(
                    self.model, **inp_cond, controlnet=self.controlnet,
                    timesteps=timesteps, guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs
                )
            else:
                x = denoise(self.model, **inp_cond, timesteps=timesteps, guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


