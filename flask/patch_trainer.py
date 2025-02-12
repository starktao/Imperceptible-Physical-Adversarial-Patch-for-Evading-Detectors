import glob
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from models.common import DetectMultiBackend

class PatchTrainer(object):
    def __init__(self, target_class):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights="yolov5s.pt", device=self.device)
        self.model.to(self.device)
        self.yolo = YOLO('yolov5s.pt')
        self.target_class = target_class
        
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0))
        self.confidence_records_11 = []
        self.confidence_records_non_11 = []
        self.best_non_target_patch = None
        self.best_non_target_conf_sum = -float('inf')
        self.best_epoch = 0

    def get_target_bbox(self, image_tensor):
        """获取目标边界框"""
        image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        results = self.yolo(image_np)
        boxes = results[0].boxes

        target_detections = []
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls == self.target_class and conf > 0.3:
                bbox = box.xyxy[0].cpu().numpy()
                target_detections.append((conf, bbox))

        if not target_detections:
            raise ValueError(f"未检测到目标类别 {self.target_class} 的物体")

        best_detection = max(target_detections, key=lambda x: x[0])
        _, bbox = best_detection

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1

        patch_size = int(bbox_width / 3)
        upper_third_y = bbox_y1 + bbox_height / 3
        x_center = (bbox_x1 + bbox_x2) / 2

        x_top_left = int(x_center - patch_size / 2)
        y_top_left = int(upper_third_y - patch_size / 2)

        _, _, H, W = image_tensor.shape
        x_top_left = max(0, min(W - patch_size, x_top_left))
        y_top_left = max(0, min(H - patch_size, y_top_left))

        return patch_size, (y_top_left, x_top_left)

    def generate_patch(self, type='gray'):
        """生成初始补丁"""
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 640, 640), 0.5, device=self.device, requires_grad=True)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 640, 640), device=self.device, requires_grad=True)
        return adv_patch_cpu

    def preprocess_image(self, image_path, img_size=640):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.type(torch.float32).to(self.device)
        return image_tensor

    def apply_patch(self, image_tensor, patch, patch_location, patch_size, epoch, img_index):
        """应用补丁到图像"""
        y, x = patch_location
        _, _, H, W = image_tensor.shape

        if epoch < 3000:
            delta_x = delta_y = patch_size_delta = 0
        elif epoch < 2000:
            delta_x = delta_y = random.randint(-2, 2)
            patch_size_delta = random.randint(-2, 2)
        else:
            delta_x = delta_y = random.randint(-4, 4)
            patch_size_delta = random.randint(-4, 4)

        y = max(0, min(H - patch_size, y + delta_y))
        x = max(0, min(W - patch_size, x + delta_x))
        patch_size = max(1, patch_size + patch_size_delta)

        patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), 
                            mode='bilinear', align_corners=True).squeeze(0)
        patch = self.gaussian_blur(patch.unsqueeze(0)).squeeze(0)

        mask = torch.zeros((1, 3, H, W), device=self.device)
        mask[:, :, y:y + patch_size, x:x + patch_size] = 1
        padded_patch = torch.zeros((1, 3, H, W), device=self.device)
        padded_patch[:, :, y:y + patch_size, x:x + patch_size] = patch

        patched_image = torch.where(mask == 1, padded_patch, image_tensor)
        return patched_image

    def detect_and_get_confidence(self, image_tensor, target_class=0):
        """检测并获取置信度"""
        # 确保不计算梯度
        with torch.no_grad():
            # 获取模型输出
            results = self.model(image_tensor)
        
        # 获取预测结果
        if isinstance(results, tuple):
            pred = results[0]  # 获取第一个元素
        else:
            pred = results
        
        # 转换为可求导的张量
        if isinstance(pred, list):
            pred = pred[0].clone().detach().requires_grad_(True)
        elif isinstance(pred, torch.Tensor):
            pred = pred.clone().detach().requires_grad_(True)
        else:
            raise ValueError(f"Unexpected prediction type: {type(pred)}")

        num_classes = 80
        obj_confs = pred[..., 4]
        cls_confs = pred[..., 5:5 + num_classes]

        class_11_confs = cls_confs[0, :, target_class]
        combined_confs = class_11_confs * obj_confs[0, :]

        max_index = torch.argmax(combined_confs)
        max_combined_conf = combined_confs[max_index]
        best_class_conf_11 = class_11_confs[max_index]
        best_obj_conf = obj_confs[0, max_index]
        
        if max_combined_conf < 0.25:
            best_class_conf_11 = max_combined_conf

        max_obj_conf_index = torch.argmax(obj_confs[0])
        best_obj_conf = obj_confs[0, max_obj_conf_index]
        best_cls_confs = cls_confs[0, max_obj_conf_index]
        best_class_conf, best_class_index = torch.max(best_cls_confs, dim=0)

        return max_combined_conf, best_class_conf, best_class_conf_11, best_class_index

    def save_image(self, tensor, path, text=None):
        """保存图像"""
        tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype('uint8')
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        if text:
            cv2.putText(tensor, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(path, tensor)

    def nps_calculator(self, patch):
        """计算非打印性得分"""
        printability_array = torch.tensor([[0.2200, 0.2200, 0.2200], 
                                         [0.5010, 0.5010, 0.5010], 
                                         [0.8670, 0.8670, 0.8670]], device=patch.device)
        diff = patch - printability_array.unsqueeze(2).unsqueeze(3)
        diff = torch.min(diff ** 2, dim=1)[0]
        diff = torch.sum(diff, dim=(1, 2))
        return torch.mean(diff)

    def total_variation(self, patch):
        """计算总变差"""
        tv_h = torch.mean(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]))
        tv_w = torch.mean(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]))
        return tv_h + tv_w

    def color_similarity_loss(self, patch, img_tensor, img_index, use_black_background=True):
        """计算颜色相似度损失"""
        _, _, H, W = img_tensor.shape
        patch_resized = F.interpolate(patch.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True)
        target_patch = torch.zeros_like(img_tensor, device=self.device) if use_black_background else img_tensor
        loss = F.mse_loss(patch_resized, target_patch)
        max_possible_loss = torch.tensor(1.0, device=self.device)
        similarity_percentage = (1 - (loss / max_possible_loss)) * 100
        return loss, similarity_percentage.item()

    def train(self, img_tensor, n_epochs=300):
        """训练过程"""
        try:
            patch_size, patch_location = self.get_target_bbox(img_tensor)
        except ValueError as e:
            raise ValueError(f"目标检测失败: {str(e)}")

        adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu.requires_grad_(True)

        optimizer = torch.optim.Adam([adv_patch_cpu], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        for epoch in tqdm(range(n_epochs), desc="Training"):
            optimizer.zero_grad()
            patched_image = self.apply_patch(img_tensor, adv_patch_cpu, patch_location, patch_size, epoch, 0)
            
            max_combined_conf, best_class_conf, best_class_conf_11, best_class_index = \
                self.detect_and_get_confidence(patched_image, self.target_class)

            loss = max_combined_conf
            nps_loss = self.nps_calculator(adv_patch_cpu)
            tv_loss = self.total_variation(adv_patch_cpu)
            color_loss, color_similar = self.color_similarity_loss(adv_patch_cpu, img_tensor, 0)

            color_loss_weight = 0 if epoch < 1000 else 20
            total_loss = loss + color_loss * color_loss_weight + tv_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            adv_patch_cpu.data.clamp_(0, 1)

            if epoch % 200 == 0:
                print(f'Epoch {epoch} - Loss: {total_loss.item()}')
                save_path = f"saved_patches/adv_patch_epoch_{epoch}.jpg"
                self.save_image(adv_patch_cpu, save_path)
                save_path = f"saved_img/patched_image_epoch_{epoch}.jpg"
                self.save_image(patched_image, save_path)

        timestamp = int(time.time())
        patch_path = f"results/patch_{timestamp}.jpg"
        patched_image_path = f"results/patched_image_{timestamp}.jpg"
        
        self.save_image(adv_patch_cpu, patch_path)
        self.save_image(patched_image, patched_image_path)
        
        return patch_path, patched_image_path 
