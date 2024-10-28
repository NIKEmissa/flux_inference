import os
import argparse
import pandas as pd
import time
from src.flux.xflux_pipeline import XFluxPipeline

class YMLDataIterator:
    def __init__(self, file_path, header_lines=4):
        self.file_path = file_path
        self.header_lines = header_lines

    def __iter__(self):
        # 打开文件，并跳过表头部分
        with open(self.file_path, 'r', encoding='utf-8') as file:
            # 跳过表头
            for _ in range(self.header_lines):
                next(file)
            
            # 迭代数据行
            for line in file:
                yield line.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='./weights/flux-dev')
    parser.add_argument('--model_type', type=str, default='flux-dev')
    parser.add_argument('--lora_path', type=str, default='./weights/LoRA/a800_10_2x2_ck20000_lora_fix_timesteps.safetensors')
    parser.add_argument('--device', type=str, default='1,3')
    parser.add_argument('--save_imgs_dir', type=str, default='./test/ablation')
    parser.add_argument('--mode', type=str, default='exp_dev')
    parser.add_argument('--attr_index', type=int, default=0)

    args = parser.parse_args()
    args.save_imgs_dir = f"{args.save_imgs_dir}/{args.mode}_1028_wq_lora"
    
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    
    pipeline = XFluxPipeline(args.base_model_path, args.model_type, [f'cuda:{i}' for i in args.device.split(',')], False, False)
    ck_name = 'dev'
    
    neg_prompt = "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, Half-body."
    pre_prompt = 'Four images arranged in a 2x2 grid.'
    post_prompt = 'Ensure that the design elements, patterns, and colors of the clothing are consistent in the four postures, and the only difference is the visual angle.'
    comp = 'simple'

    attr_yml_files = [
                    None,
                    './data/yml/neckline.yml',
                    './data/yml/collar.yml',
                    './data/yml/cuff.yml',
                    './data/yml/front_closure_style.yml',
                    './data/yml/garment_closure.yml',
                    './data/yml/garment_length.yml',
                    './data/yml/pocket.yml',
                    './data/yml/sleeve_length.yml',
                    './data/yml/shirt_hem.yml',
                    './data/yml/shoulder.yml',
                    './data/yml/silhouette.yml',
                    './data/yml/sleeve.yml']

    lora_path = './weights/lora/model12/model12_ck14000.safetensors'
    lora_path = './weights/lora/wangqiang_base_prompt/fashion_lora (9).safetensors'
    resolutions = [[1024, 1024], [512, 512]]

    for attr_yml_file in [attr_yml_files[args.attr_index]]:
        attr_cate = attr_yml_file.split('/')[-1].split('.')[0]

        for attr in YMLDataIterator(attr_yml_file, 4):
            attr = attr.replace('/', ' ')
            if attr.lower() == 'other': continue
            prompt = f"In a clean and minimal setting, a full-body front view of a Chinese girl model, is presented.The model is wearing a {attr} T-shirt."

            for p_index, prompt in enumerate([prompt, f"{pre_prompt} {prompt} {post_prompt}"]):
                print(prompt)

                if args.mode == 'exp_dev':              
                    guidances = [3.5] if p_index == 0 else [2.5]
                    img_seq_lens = [1.15] if p_index == 0 else [0.4]
                    seeds = [
                        5456756856,
                        9876543210,
                        192837465,
                        765432198,
                        876543219,
                        654321987,
                        123098765,
                        456789123,
                        789456123,
                        147852369
                    ]
                    lora_weights = ['0', '0.9']
                    for seed in seeds:
                        for guidance in guidances:
                            for img_seq_len in img_seq_lens:    
                                mu = img_seq_len
                                for lora_weight in lora_weights:
                                    for resolution in resolutions:
                                        if lora_weight == '0.9':
                                            ck_name = lora_path.split('/')[-1].split('.')[0]
                                            # f_width = 1024 * 2
                                            # f_height = 1024 * 2
                                        else:
                                            ck_name = 'dev'
                                            # f_width = 768
                                            # f_height = 768
                                        f_width = resolution[0]
                                        f_height = resolution[1]

                                        if p_index == 1:
                                            f_width = f_width * 2
                                            f_height = f_height * 2

                                        lora_weight = float(lora_weight)
                                        pipeline.set_lora(lora_path, lora_weight=lora_weight)  

                                        # 增加计时
                                        start_time = time.perf_counter()
                                        print("attr:{} seed:{}, guidance:{}, mu:{}, f_width:{}, f_height:{}, ck_name:{}, lora_weight:{}, attr_cate:{}, prompt:{}".format(attr, seed, guidance, mu, f_width, f_height, ck_name, lora_weight, attr_cate, prompt))
                                        out_img = pipeline(
                                            prompt=prompt,
                                            controlnet_image=None,
                                            width=f_width,
                                            height=f_height,
                                            guidance=guidance,
                                            num_steps=50,
                                            true_gs=4,
                                            neg_prompt=neg_prompt,
                                            timestep_to_start_cfg=200,
                                            seed=seed,
                                            callback=None,
                                            cb_infos=None,
                                            progress_len=95,
                                            img_seq_len=mu
                                        )

                                        prompt_tag = "1grid" if p_index == 0 else "4grid"
                                        outfile = f"{args.save_imgs_dir}/ck_name@{ck_name}_lora_weight@{lora_weight}_promptTag@{prompt_tag}_attr_cate@{attr_cate}_attr@{attr}_seed@{seed}_guide@{guidance}_mu@{mu}_resolution@{f_height}*{f_width}_comp@{comp}.jpg"
                                        out_img.save(outfile, quality=95)
                                        end_time = time.perf_counter()

                                        elapsed_time_ms = (end_time - start_time) * 1000
                                        print(f"Elapsed time: {elapsed_time_ms} ms")
