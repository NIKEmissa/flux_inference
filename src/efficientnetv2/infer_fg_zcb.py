import torch
from torchvision import transforms
from PIL import Image
from src.efficientnetv2.efficientnet_v2 import get_efficientnet_v2
import os
import numpy as np

import pdb

class human_pose():
    def __init__(self, weights_root, model_name="efficientnet_v2_s_in21k", pretrained=False, num_classes=4, device="cuda:0"):
        self.device = device
        self.model = get_efficientnet_v2(model_name, pretrained, num_classes)
        
        base_model_path = os.path.join(weights_root, 'pose/human_pose.ckpt')
        state_dict = torch.load(base_model_path)['state_dict']
        corrected_state_dict = self.remove_module_prefix(state_dict)
        self.model.load_state_dict(corrected_state_dict,strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                           ])
        
        self.label_dict = { 0:"背面", 1:"正面", 2:"侧面", 3:"坐着"}

    def remove_module_prefix(self, state_dict):
        """从权重键中移除'model.'前缀"""
        new_state_dict = {}
        for k, v in state_dict.items():
            # if k.startswith('model.'):
            if k.startswith('EMA_model.'):
                # 移除前缀
                # print(k)
                new_key = k[10:]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    @torch.no_grad()
    def infer(self, pil_image):
        image = self.transform(pil_image).unsqueeze(0)  # 增加批次维度
        outputs = self.model(image.to(self.device))
        _, preds = torch.max(outputs, 1)
        res = self.label_dict[preds.item()]

        return res
