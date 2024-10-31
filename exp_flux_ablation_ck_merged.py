import os
import argparse
import time
import logging
from datetime import datetime
from itertools import product
from src.flux.xflux_pipeline import XFluxPipeline

import pandas as pd

class YMLDataIterator:
    """迭代器，用于遍历 YML 和 CSV 文件，跳过指定的表头行数。"""
    def __init__(self, file_path, header_lines=4, start_row=None, end_row=None):
        self.file_path = file_path
        self.header_lines = header_lines
        self.start_row = start_row
        self.end_row = end_row

    def __iter__(self):
        if self.file_path.endswith('.yml'):
            # 打开 YML 文件并跳过表头行
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for _ in range(self.header_lines):
                    next(file)
                for line in file:
                    yield line.strip()
        elif self.file_path.endswith('.csv'):
            # 读取 CSV 文件到 DataFrame
            df = pd.read_csv(self.file_path)

            # 确定合法的起始和结束行
            start_row = max(0, self.start_row) if self.start_row is not None else 0
            end_row = min(len(df), self.end_row) if self.end_row is not None else len(df)

            # 遍历指定范围内的每一行，组合 Keyword 和 category 列
            for idx in range(start_row, end_row):
                row = df.iloc[idx]
                keyword = row['Keyword']
                category = row['category']
                yield f"{keyword}, which is the attribute of {category}"
        else:
            raise ValueError("不支持的文件格式")

def generate_image(pipeline, prompt_info, args):
    """生成图像并保存，同时记录耗时和可能的错误。"""
    try:
        pipeline.set_lora(prompt_info['lora_path'], lora_weight=prompt_info['lora_weight'])

        start_time = time.perf_counter()
        logger.info(
            "属性: %s, 种子: %s, 引导: %s, 序列长度: %s, 宽度: %s, 高度: %s, 检查点名称: %s, LoRA 权重: %s, 属性类别: %s, 提示词: %s",
            prompt_info['attr'], prompt_info['seed'], prompt_info['guidance'], prompt_info['img_seq_len'],
            prompt_info['f_width'], prompt_info['f_height'], prompt_info['ck_name'], prompt_info['lora_weight'],
            prompt_info['attr_cate'], prompt_info['prompt']
        )
        
        out_img = pipeline(
            prompt=prompt_info['prompt'],
            controlnet_image=None,
            width=prompt_info['f_width'],
            height=prompt_info['f_height'],
            guidance=prompt_info['guidance'],
            num_steps=args.num_steps,
            true_gs=args.true_gs,
            neg_prompt=prompt_info['neg_prompt'],
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            seed=prompt_info['seed'],
            callback=None,
            cb_infos=None,
            progress_len=args.progress_len,
            img_seq_len=prompt_info['img_seq_len']
        )

        # 定义辅助函数来处理文件名中的变量
        def sanitize_filename_component(component):
            """替换字符串中的 '_' 和 '/' 为 '-'，并将组件转换为字符串。"""
            return str(component).replace('_', '-').replace('/', '-')

        filename = (
            f"ck_name@{sanitize_filename_component(prompt_info['ck_name'])}"
            f"_lora_weight@{sanitize_filename_component(prompt_info['lora_weight'])}"
            f"_promptTag@{sanitize_filename_component(prompt_info['prompt_tag'])}"
            f"_attr_cate@{sanitize_filename_component(prompt_info['attr_cate'])}"
            f"_attr@{sanitize_filename_component(prompt_info['attr'])}"
            f"_seed@{sanitize_filename_component(prompt_info['seed'])}"
            f"_guide@{sanitize_filename_component(prompt_info['guidance'])}"
            f"_img_seq_len@{sanitize_filename_component(prompt_info['img_seq_len'])}"
            f"_resolution@{sanitize_filename_component(prompt_info['f_height'])}"
            f"*{sanitize_filename_component(prompt_info['f_width'])}"
            f"_comp@{sanitize_filename_component(prompt_info['comp'])}.jpg"
        )

        outfile = os.path.join(args.save_imgs_dir, filename)
        out_img.save(outfile, quality=95)
        end_time = time.perf_counter()

        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"耗时: {elapsed_time_ms} 毫秒")
    except Exception as e:
        logger.error(f"生成图像时发生错误：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用 XFluxPipeline 生成图像的脚本。')
    parser.add_argument('--base_model_path', type=str, default='./weights/flux-dev', help='基础模型的路径。')
    parser.add_argument('--model_type', type=str, default='flux-dev', help='模型类型。')
    parser.add_argument('--lora_path', type=str, default='./weights/LoRA/a800_10_2x2_ck20000_lora_fix_timesteps.safetensors', help='LoRA 权重的路径。')
    parser.add_argument('--device', type=str, default='1,3', help='逗号分隔的 GPU 设备 ID 列表。')
    parser.add_argument('--save_imgs_dir', type=str, default='./test/ablation', help='保存生成图像的目录。')
    parser.add_argument('--mode', type=str, default='exp_dev', help='运行模式。')
    parser.add_argument('--attr_index', type=int, default=0, help='要使用的属性 YML 文件的索引。')
    parser.add_argument('--num_steps', type=int, default=50, help='生成图像的步数。')
    parser.add_argument('--true_gs', type=int, default=4, help='真实的引导缩放。')
    parser.add_argument('--timestep_to_start_cfg', type=int, default=200, help='开始 CFG 的时间步。')
    parser.add_argument('--progress_len', type=int, default=95, help='进度长度。')
    parser.add_argument('--save_log_dir', type=str, default='./log')
    parser.add_argument('--save_log_name', type=str, default='log.txt')
    parser.add_argument('--start_row', type=int, help='CSV 文件遍历的起始行。')
    parser.add_argument('--end_row', type=int, help='CSV 文件遍历的结束行。')
    args = parser.parse_args()

    tailfix = '1030_wq_lora_debug'
    args.save_imgs_dir = os.path.join(args.save_imgs_dir, f"{args.mode}_{tailfix}")
    os.makedirs(args.save_imgs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_log_dir = os.path.join(args.save_log_dir, f"{args.mode}_{timestamp}_{tailfix}")
    os.makedirs(args.save_log_dir, exist_ok=True)

    # 定义日志文件的路径
    log_file = os.path.join(args.save_log_dir, 'app.log')

    # 配置日志，日志将保存到指定的文件中，并输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)    

    pipeline = XFluxPipeline(
        args.base_model_path,
        args.model_type,
        [f'cuda:{i}' for i in args.device.split(',')],
        False,
        False
    )

    neg_prompt = "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, Half-body."
    comp = 'simple'

    DATA_DIR = './data/yml'
    attr_yml_files = [
        None,
        os.path.join(DATA_DIR, 'neckline.yml'),
        os.path.join(DATA_DIR, 'collar.yml'),
        os.path.join(DATA_DIR, 'cuff.yml'),
        os.path.join(DATA_DIR, 'front_closure_style.yml'),
        os.path.join(DATA_DIR, 'garment_closure.yml'),
        os.path.join(DATA_DIR, 'garment_length.yml'),
        os.path.join(DATA_DIR, 'pocket.yml'),
        os.path.join(DATA_DIR, 'sleeve_length.yml'),
        os.path.join(DATA_DIR, 'shirt_hem.yml'),
        os.path.join(DATA_DIR, 'shoulder.yml'),
        os.path.join(DATA_DIR, 'silhouette.yml'),
        os.path.join(DATA_DIR, 'sleeve.yml'),
        "/data/xd/MyCode/Misc/filter_labels2.csv"
    ]

    # 验证 attr_index
    if args.attr_index < 1 or args.attr_index >= len(attr_yml_files):
        parser.error(f"attr_index 必须在 1 和 {len(attr_yml_files) - 1} 之间。")

    attr_yml_file = attr_yml_files[args.attr_index]
    if attr_yml_file is None:
        logger.error("给定的 attr_index 没有指定 attr_yml_file。")
        exit(1)
    attr_cate = os.path.basename(attr_yml_file).split('.')[0]

    lora_path = './weights/lora/wangqiang_base_prompt/fashion_lora (9).safetensors'
    lora_path = './weights/lora/model12/model12_ck14000.safetensors'
    resolutions = [[1024, 1024], [512, 512]]

    # 定义与 prompt_index 相关的参数
    prompt_configs = [
        {
            'prompt_tag': '1grid',
            'pre_prompt': 'single photo.e-commerce photo 2024.',
            'post_prompt': '',
            'res_multiplier_width': 1,
            'res_multiplier_height': 1
        },
        {
            'prompt_tag': '4grid',
            'pre_prompt': 'The photographs are arranged in a 2 by 2 grid format.',
            'post_prompt': 'Ensure that the design elements, patterns, and colors of the clothing are consistent in the four postures, and the only difference is the visual angle.',
            'res_multiplier_width': 2,
            'res_multiplier_height': 2
        }
    ]

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
    lora_weights = [0.0, 0.9]

    for attr in YMLDataIterator(attr_yml_file, 4, args.start_row, args.end_row):
        attr = attr.replace('/', ' ')
        if attr.lower() == 'other':
            continue

        base_prompt = f"In a clean and minimal setting, a full-body front view of a Chinese girl model is presented. The model is wearing a {attr} T-shirt."

        for prompt_config in prompt_configs:
            prompt = f"{prompt_config['pre_prompt']} {base_prompt} {prompt_config['post_prompt']}".strip()
            logger.info(prompt)

            if args.mode == 'exp_dev':
                parameter_combinations = product(
                    seeds,
                    lora_weights,
                    resolutions
                )

                for seed, lora_weight, resolution in parameter_combinations:
                    # 设置 guidance 和 img_seq_len 根据 lora_weight
                    if lora_weight == 0.9:
                        guidance = 4.0
                        img_seq_len = 1.5
                        ck_name = os.path.basename(lora_path).split('.')[0]
                    else:
                        guidance = 3.5
                        img_seq_len = 1.15
                        ck_name = 'dev'

                    base_width, base_height = resolution
                    f_width = base_width * prompt_config['res_multiplier_width']
                    f_height = base_height * prompt_config['res_multiplier_height']

                    prompt_info = {
                        'pipeline': pipeline,
                        'prompt': prompt,
                        'neg_prompt': neg_prompt,
                        'seed': seed,
                        'guidance': guidance,
                        'img_seq_len': img_seq_len,
                        'lora_path': lora_path,
                        'lora_weight': lora_weight,
                        'f_width': f_width,
                        'f_height': f_height,
                        'ck_name': ck_name,
                        'prompt_tag': prompt_config['prompt_tag'],
                        'attr_cate': attr_cate,
                        'attr': attr,
                        'comp': comp
                    }

                    generate_image(
                        pipeline=pipeline,
                        prompt_info=prompt_info,
                        args=args
                    )
