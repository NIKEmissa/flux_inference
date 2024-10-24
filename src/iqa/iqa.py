import os
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from easydict import EasyDict as edict

from src.iqa.iqa_utils import get_network
from src.iqa.utils.general import check_img_size, non_max_suppression, scale_boxes
from src.iqa.models.common import DetectMultiBackend
from src.iqa.utils.torch_utils import select_device
from src.iqa.utils.augmentations import letterbox
import pdb

default_conf = {
    'bodydet':{
        "data": "bodypart.yaml",
        "weights": "aigc_iqa/best.pt",
        "half": 0,
        "conf_thres": 0.7,
        "iou_thres": 0.45,
        "imgsz": [640,640]
    },
    'face': {
        'backbone': 'resnet18',
        'cls': 3,
        'weights': 'aigc_iqa/face_v2.pth',
        'thres': 0.5
    },
    'human': {
        'backbone': 'resnet18',
        'cls': 5,
        'weights': 'aigc_iqa/human_v2.pth',
        'thres': 0.5
    },
    'hand': {
        'backbone': 'resnet18',
        'cls': 3,
        'weights': 'aigc_iqa/hand_v2.pth',
        'thres': 0.5
    },
    
    
    'foot': {
        'backbone': 'resnet18',
        'cls': 1,
        'weights': 'aigc_iqa/foot_v2.pth',
        'thres': 0.5
    }
}


class AigcQuality():
    def __init__(self, model_dir=None, imgsize=448):
        self.aigcqlt_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imgsize, imgsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.conf = edict(default_conf)
        try:
            # 检测模型加载
            gpu_num = torch.cuda.current_device()
            self.device = torch.device('cuda:0') #select_device(str(gpu_num))
            self.bodydet_conf_thres = self.conf.bodydet.conf_thres
            self.bodydet_iou_thres = self.conf.bodydet.iou_thres

            model_path = os.path.join(model_dir, self.conf.bodydet.weights)
            if not os.path.exists(model_path):
                print('bodydet model path does not exists:', model_path)
            self.bodydet_model = DetectMultiBackend(model_path, device=self.device, dnn=False, data=self.conf.bodydet.data, fp16=self.conf.bodydet.half)
            self.bodydet_model.warmup(imgsz=(1, 3, *self.conf.bodydet.imgsz))  # warmup            
            print('bodydet model init')            

            # 头肩部质量识别模型加载
            self.face_iqa = get_network(self.conf.face.backbone, self.conf.face.cls)
            face_modelpath = os.path.join(model_dir, self.conf.face.weights)
            face_ckpt = torch.load(face_modelpath, map_location=torch.device('cpu'))
            self.face_iqa.load_state_dict({k.replace('module.', ''): v for k, v in face_ckpt['model'].items()})
            self.face_iqa.eval()
            # self.face_iqa.cuda()
            self.face_iqa = self.face_iqa.to(self.device)
            print('face area iqa model init')            
            # 人体区域质量识别模型加载
            self.human_iqa = get_network(self.conf.human.backbone, self.conf.human.cls)
            human_modelpath = os.path.join(model_dir, self.conf.human.weights)
            human_ckpt = torch.load(human_modelpath, map_location=torch.device('cpu'))
            self.human_iqa.load_state_dict({k.replace('module.', ''): v for k, v in human_ckpt['model'].items()})
            self.human_iqa.eval()
            # self.human_iqa.cuda()
            self.human_iqa = self.human_iqa.to(self.device)
            print('human area iqa model init')
            
            # ：手部区域质量识别模型加载
            self.hand_iqa = get_network(self.conf.hand.backbone, self.conf.hand.cls)
            hand_modelpath = os.path.join(model_dir, self.conf.hand.weights)
            hand_ckpt = torch.load(hand_modelpath, map_location=torch.device('cpu'))
            self.hand_iqa.load_state_dict({k.replace('module.', ''): v for k, v in hand_ckpt['model'].items()})
            self.hand_iqa.eval()
            # self.hand_iqa.cuda()
            self.hand_iqa = self.hand_iqa.to(self.device)
            print('hand area iqa model init')
            
            # 脚部区域质量识别模型加载
            self.foot_iqa = get_network(self.conf.foot.backbone, self.conf.foot.cls)
            
            foot_modelpath = os.path.join(model_dir, self.conf.foot.weights)
            foot_ckpt = torch.load(foot_modelpath, map_location=torch.device('cpu'))
            self.foot_iqa.load_state_dict({k.replace('module.', ''): v for k, v in foot_ckpt['model'].items()})
            self.foot_iqa.eval()
            # self.foot_iqa.cuda()
            self.foot_iqa = self.foot_iqa.to(self.device)
            print('foot area iqa model init')
        except Exception as e:
            print(e)
            raise Exception('AigcQuality model init error...')
        print('All AigcQuality model has been loaded...')

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    def extract_roi(self, img, bbox=None, roi_type='face', imgsize=512):
        if np.all(bbox == 0): return None
        H, W, C = img.shape
        region = None
        if roi_type == 'human':
            if H == W:
                return cv2.resize(img, (imgsize, imgsize), interpolation=cv2.INTER_AREA)
            top = max(W - H, 0) // 2
            left = max(H - W, 0) // 2
            bottom = max(W - H, 0) - top
            right = max(H - W, 0) - left
            img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, 0)
            region = cv2.resize(img_pad, (imgsize, imgsize), interpolation=cv2.INTER_AREA)
        elif roi_type == 'face':
            xc, yc, w, h = bbox
            face_w, face_h = int(W * w), int(H * h)
            Pad = max(face_w, face_h) * 2
            img_pad = cv2.copyMakeBorder(img, Pad, Pad, Pad, Pad, 0)
            pad = int(max(face_w, face_h) * 1.5)
            top = int(yc * H) - pad + Pad
            left = int(xc * W) - pad + Pad
            bottom = int(yc * H) + pad + Pad
            right = int(xc * W) + pad + Pad
            pad_ = max(bottom - top, right - left)
            region = img_pad[top:top + pad_, left:left + pad_]        
        elif roi_type == 'foot' or roi_type == 'hand':
            xc, yc, w, h = bbox
            box_w, box_h = int(W * w), int(H * h)
            Pad = max(box_w, box_h) * 1
            img_pad = cv2.copyMakeBorder(img, Pad, Pad, Pad, Pad, 0)
            pad = int(max(box_w, box_h) * 0.75)
            top = int(yc * H) - pad + Pad
            left = int(xc * W) - pad + Pad
            bottom = int(yc * H) + pad + Pad
            right = int(xc * W) + pad + Pad
            pad_ = max(bottom - top, right - left)
            region = img_pad[top:top + pad_, left:left + pad_]
        return region
    
    def forward(self, img, task_type='face'):
        tensor = self.aigcqlt_transforms(img).unsqueeze(0).to(self.device)
        pred = None
        if task_type == 'face':
            pred = self.face_iqa(tensor)[0].data.cpu().numpy()
        elif task_type == 'human':
            pred = self.human_iqa(tensor)[0].data.cpu().numpy()
        elif task_type == 'hand':
            pred = self.hand_iqa(tensor)[0].data.cpu().numpy()
        elif task_type == 'foot':
            pred = self.foot_iqa(tensor)[0].data.cpu().numpy()
        return pred
    
    def fuse_score(self, rois_score, weights=[1.0] * 11):
        res = 1 - np.mean(rois_score)
        return res

    def infer(self, img):
        if type(img) == str:
            img = cv2.imread(img)
        elif type(img) == Image.Image:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        src_img = img.copy()
        score = 1.0
        try:
            stride, pt = self.bodydet_model.stride, self.bodydet_model.pt
            imgsz = check_img_size(self.conf.bodydet.imgsz, s=stride)
            img = letterbox(img, imgsz, stride=stride, auto=pt)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.conf.bodydet.half else img.float()  # uint8 to fp16/32
            img /= 255
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            pred = self.bodydet_model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.bodydet_conf_thres, self.bodydet_iou_thres, None, False, max_det=100)
            # Process predictions
            rois = np.zeros((6, 6), dtype=np.float32)
            rois_score = [0] * 11
            gn = torch.tensor(src_img.shape)[[1, 0, 1, 0]]
            for i, det in enumerate(pred):  # per image
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], src_img.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        conf = conf.data.cpu().numpy()
                        cls = cls.data.cpu().numpy()
                        idx = int(cls)
                        xywh = (self.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        if conf > rois[idx][4]:
                            rois[idx] = xywh + [conf, idx]
                    
            # 头肩部
            face_roi = self.extract_roi(src_img, bbox=rois[1][:4], roi_type='face')            
            if face_roi is not None:
                # cv2.imwrite('face_roi.jpg', face_roi)
                face_score = self.forward(face_roi, task_type='face')
                rois_score[:3] = face_score
            # 人体
            human_roi = self.extract_roi(src_img, roi_type='human')
            if human_roi is not None:
                # cv2.imwrite('human_roi.jpg', human_roi)
                human_score = self.forward(human_roi, task_type='human')
                rois_score[3:5] = human_score[:2]
            # 手部: 左手/右手
            lhand_roi = self.extract_roi(src_img, bbox=rois[2][:4], roi_type='hand')
            if lhand_roi is not None:
                # cv2.imwrite('lhand_roi.jpg', lhand_roi)
                lhand_score = self.forward(lhand_roi, task_type='hand')
                rois_score[5:7] = lhand_score[:2]
            rhand_roi = self.extract_roi(src_img, bbox=rois[3][:4], roi_type='hand')
            if rhand_roi is not None:
                # cv2.imwrite('rhand_roi.jpg', rhand_roi)
                rhand_score = self.forward(rhand_roi, task_type='hand')
                rois_score[7:9] = rhand_score[:2]
            # 脚部: 左脚/右脚
            lfoot_roi = self.extract_roi(src_img, bbox=rois[4][:4], roi_type='foot')
            if lfoot_roi is not None:
                # cv2.imwrite('lfoot_roi.jpg', lfoot_roi)
                lfoot_score = self.forward(lfoot_roi, task_type='foot')
                rois_score[9] = lfoot_score[0]
            rfoot_roi = self.extract_roi(src_img, bbox=rois[5][:4], roi_type='foot')
            if rfoot_roi is not None:
                # cv2.imwrite('rfoot_roi.jpg', rfoot_roi)
                rfoot_score = self.forward(rfoot_roi, task_type='foot')
                rois_score[10] = rfoot_score[0]

            # 融合分数
            # print(rois_score)
            score = self.fuse_score(rois_score)
        
        except Exception as e:
            print('AigcQuality infer error:', e)
        
        return score
    
if __name__ == '__main__':
    aigc_iqa = AigcQuality(model_dir='/data/sunhuanrong/projects/diffusersProj/xfluxProj/src/iqa')# 运行文件路径
    # img = cv2.imread('haotu.jpg')
    img = cv2.imread('duoshou.jpg')
    # img = cv2.imread('weibeng.jpg')


    score = aigc_iqa.infer(img)
    print(score)
