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
    parser.add_argument('--device', type=str, default='5,7')
    parser.add_argument('--save_imgs_dir', type=str, default='./test/ablation')
    parser.add_argument('--mode', type=str, default='exp_dev')
    parser.add_argument('--attr_index', type=int, default=0)

    args = parser.parse_args()
    args.save_imgs_dir = f"{args.save_imgs_dir}/{args.mode}_1027"
    
    os.makedirs(args.save_imgs_dir, exist_ok=True)
    
    pipeline = XFluxPipeline(args.base_model_path, args.model_type, [f'cuda:{i}' for i in args.device.split(',')], False, False)
    ck_name = 'dev'
    
    neg_prompt = "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, Half-body."
    pre_prompt = 'Four images arranged in a 2x2 grid.'
    post_prompt = 'Ensure that the design elements, patterns, and colors of the clothing are consistent in the four postures, and the only difference is the visual angle.'
    comp = 'complicated'

    attr_yml_files = ['./data/yml/neckline.yml',
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

        for attr_yml_file in [attr_yml_files[args.attr_index]]:
            attr_cate = attr_yml_file.split('/')[-1].split('.')[0] if int(seed) != 0 else 'original'

            for attr in YMLDataIterator(attr_yml_file, 4):
                attr = attr.replace('/', ' ')
                
                shirt_hem = 'asymmetrical hemline'
                shoulder = 'regular shoulders'
                neckline = 'square neckline'
                collar = 'narrow, flat collar'
                sleeve = 'long sleeves'
                sleeve_length = 'long'
                cuff = 'slightly flared cuffs'
                front_closure_style = 'pullover style'
                garment_closure = 'easy wear'
                garment_length = 'falls just below the hips'
                pocket = 'no pockets'
                silhouette = 'dynamic silhouette'

                if attr.lower() == 'other': continue
                if attr_cate == 'shirt_hem':
                    shirt_hem = attr
                elif attr_cate == 'shoulder':
                    shoulder = attr
                elif attr_cate == 'neckline':
                    neckline = attr
                elif attr_cate == 'collar':
                    collar = attr
                elif attr_cate == 'sleeve':
                    sleeve = attr
                elif attr_cate == 'sleeve_length':
                    sleeve_length = attr
                elif attr_cate == 'cuff':
                    cuff = attr                                
                elif attr_cate == 'front_closure_style':
                    front_closure_style = attr
                elif attr_cate == 'garment_closure':
                    garment_closure = attr
                elif attr_cate == 'silhouette':
                    silhouette = attr
                elif attr_cate == 'garment_length':
                    garment_length = attr
                elif attr_cate == 'pocket':
                    pocket = attr

                # prompt = f"In a clean and minimal setting, a full-body front view of a Chinese girl model, is presented.The model is wearing a {attr} T-shirt."
                # prompt = f"""
                #     The image presents an indoor studio setting with a minimalist background. The backdrop is solid white, enhancing the focus on the model and the outfit without any distractions from additional props or accessories. The model is wearing a fitted long-sleeve top with an asymmetrical hemline. The top features regular shoulders and a square neckline, which is notably wide, accentuating the collarbone. The sleeves are long and slightly flared at the cuffs, adding a touch of elegance to the overall design. The side seams include gathered details, providing a unique texture and fit, while the hemline is angled, creating a dynamic silhouette. The garment is made from a soft stretch fabric, ensuring comfort and ease of movement. This fabric has a smooth texture with a slight sheen, reflecting light softly. The primary color is white, while the graphic print on the front features a rustic brown with vintage lettering that reads ""Cafe Racer Vintage,"" adorned with stars and a dreamy slogan, adding a playful element to the design. The model is a South Asian woman, approximately 20 years old, with long, wavy dark hair. Her skin tone is medium, and she possesses a petite stature. She stands in a slight pose, with her left arm extended elegantly, showcasing the top's details. Her expression is thoughtful, enhancing the overall aesthetic of the look. The makeup is natural, featuring a light foundation and soft pink lip color, complementing the outfit without overpowering it. The lighting is soft and diffused, emphasizing the model's features and the top's design without harsh shadows. The composition centers on the model, ensuring the garment remains the focal point, showcasing the graphic print, unique neckline, and flared sleeves effectively. Overall, this presentation highlights the top's key features: the asymmetrical design, gathered side details, and the vintage graphic print.            
                # """
                # prompt = f"The image presents an indoor studio setting with a minimalist background. The backdrop is solid white, enhancing the focus on the model and the outfit without any distractions from additional props or accessories. The model is wearing a fitted long-sleeve top with an {shirt_hem} **(shirt_hem)**. The top features {shoulder} **(shoulder)** and a {neckline} **(neckline)**. The top also includes a narrow, {collar} **(collar)** that adds subtle structure to the neckline. The {sleeve} sleeves **(sleeve)** are {sleeve_length} **(sleeve_length)** and is a {cuff} **(cuff)**, adding a touch of elegance to the overall design. The front of the garment has no visible buttons or zippers, indicating a {front_closure_style} front closure **(front_closure_style)** for {garment_closure} **(garment_closure)**. The side seams include gathered details, providing a unique texture and fit, while the hemline is angled, creating a {silhouette} **(silhouette)**. The garment length **(garment_length)** is {garment_length}, offering a comfortable and flattering fit, contributing to the hip-length cut. There are {pocket} **(pocket)** on the garment, maintaining a sleek and minimalist appearance. The garment is made from a soft stretch fabric, ensuring comfort and ease of movement. This fabric has a smooth texture with a slight sheen, reflecting light softly. The primary color is white, while the graphic print on the front features a rustic brown with vintage lettering that reads 'Cafe Racer Vintage', adorned with stars and a dreamy slogan, adding a playful element to the design. The model is a South Asian woman, approximately 20 years old, with long, wavy dark hair. Her skin tone is medium, and she possesses a petite stature. She stands in a slight pose, with her left arm extended elegantly, showcasing the top's details. Her expression is thoughtful, enhancing the overall aesthetic of the look. The makeup is natural, featuring a light foundation and soft pink lip color, complementing the outfit without overpowering it. The lighting is soft and diffused, emphasizing the model's features and the top's design without harsh shadows. The composition centers on the model, ensuring the garment remains the focal point, showcasing the graphic print, unique neckline, and flared sleeves effectively. Overall, this presentation highlights the top's key features: the asymmetrical design, gathered side details, vintage graphic print."
                prompt = f"The image presents an indoor studio setting with a minimalist background. The backdrop is solid white, enhancing the focus on the model and the outfit without any distractions from additional props or accessories. The model is wearing a t-shirt with an {shirt_hem} hem of the top. The top features {shoulder} shoulder and a {neckline} neckline. The top also includes a {collar} collar. The {sleeve} sleeves are with {sleeve_length} sleeve_length and is a {cuff} cuff, adding a touch of elegance to the overall design. The front of the garment is indicating a {front_closure_style} front closure for {garment_closure} garment_closure. The cloth is a {silhouette} silhouette. The garment length is {garment_length}. There are {pocket} pocket on the garment, maintaining a sleek and minimalist appearance. The garment is made from a soft stretch fabric, ensuring comfort and ease of movement. This fabric has a smooth texture with a slight sheen, reflecting light softly. The primary color is white, while the graphic print on the front features a rustic brown with vintage lettering that reads 'Cafe Racer Vintage', adorned with stars and a dreamy slogan, adding a playful element to the design. The model is a South Asian woman, approximately 20 years old, with long, wavy dark hair. Her skin tone is medium, and she possesses a petite stature. She stands in a slight pose, with her left arm extended elegantly, showcasing the top's details. Her expression is thoughtful, enhancing the overall aesthetic of the look. The makeup is natural, featuring a light foundation and soft pink lip color, complementing the outfit without overpowering it. The lighting is soft and diffused, emphasizing the model's features and the top's design without harsh shadows. The composition centers on the model, ensuring the garment remains the focal point, showcasing the graphic print, unique neckline, and flared sleeves effectively. "

                # for p_index, prompt in enumerate([prompt, f"{pre_prompt} {prompt} {post_prompt}"]):
                for p_index, prompt in enumerate([f"{pre_prompt} {prompt} {post_prompt}"]):

                    if args.mode == 'exp_dev':              
                        guidances = [3.5] if p_index == 0 else [2.5]
                        img_seq_lens = [1.15] if p_index == 0 else [0.4]
                        width, height = 768, 768

                        # lora_weights = ['0', '0.9']
                        lora_weights = ['0.9']
                        for guidance in guidances:
                            for img_seq_len in img_seq_lens:    
                                mu = img_seq_len
                                for lora_weight in lora_weights:
                                    if lora_weight == '0.9':
                                        ck_name = lora_path.split('/')[-1].split('.')[0]
                                        f_width = 1024 * 2
                                        f_height = 1024 * 2
                                    else:
                                        ck_name = 'dev'
                                        f_width = 768
                                        f_height = 768

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
