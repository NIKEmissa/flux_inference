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

    args = parser.parse_args()
    args.save_imgs_dir = f"{args.save_imgs_dir}/{args.mode}_1018"
    
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    
    pipeline = XFluxPipeline(args.base_model_path, args.model_type, [f'cuda:{i}' for i in args.device.split(',')], False, False)
    ck_name = 'dev'
    
    neg_prompt = "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, Half-body."
    pre_prompt = 'Four images arranged in a 2x2 grid.'
    post_prompt = 'Ensure that the design elements, patterns, and colors of the clothing are consistent in the four postures, and the only difference is the visual angle.'

    attr_yml_files = ['/data/xd/MyCode/Project/exp_vs/data/yml/neckline.yml']

    for attr_yml_file in attr_yml_files:

        for attr in YMLDataIterator(attr_yml_file):
            if attr.lower() == 'other': continue
            prompt = f"In a clean and minimal setting, a full-body front view of a Chinese girl model, with arms fully extended horizontally to form a 90-degree angle with the torso, is presented.The model is wearing a {attr} T-shirt."

            for p_index, prompt in enumerate([prompt, f"{pre_prompt} {prompt} {post_prompt}"]):
                print(prompt)

                if args.mode == 'exp_dev':              
                    guidances = [3.5]
                    img_seq_lens = [1.15]
                    width, height = 768, 1024            
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
                    for seed in seeds:
                        for guidance in guidances:
                            for img_seq_len in img_seq_lens:    
                                # 增加计时
                                start_time = time.perf_counter()
                                print("attr:{} seed:{}, guidance:{}, img_seq_len:{}".format(attr, seed, guidance, img_seq_len))
                                out_img = pipeline(
                                    prompt=prompt,
                                    controlnet_image=None,
                                    width=width * 2,
                                    height=height * 2,
                                    guidance=guidance,
                                    num_steps=50,
                                    true_gs=4,
                                    neg_prompt=neg_prompt,
                                    timestep_to_start_cfg=200,
                                    seed=seed,
                                    callback=None,
                                    cb_infos=None,
                                    progress_len=95,
                                    img_seq_len=img_seq_len
                                )

                                prompt_tag = "1grid" if p_index == 0 else "4grid"
                                outfile = f"{args.save_imgs_dir}/ck_name@{ck_name}_promptTag@{prompt_tag}_attr@{attr}_seed@{seed}_guide@{guidance}_mu@{img_seq_len}.jpg"
                                out_img.save(outfile, quality=95)
                                end_time = time.perf_counter()

                                elapsed_time_ms = (end_time - start_time) * 1000
                                print(f"Elapsed time: {elapsed_time_ms} ms")
