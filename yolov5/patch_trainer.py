import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from models.common import DetectMultiBackend
from utils.patch_utils import PatchUtils

class PatchTrainer(object):
    def __init__(self, mode, label_folder, image_width, image_height):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights="yolov5s.pt", device=self.device)
        self.model.to(self.device)
        
        # 从标签文件中获取补丁大小和位置
        self.patch_sizes, self.patch_locations = self._load_patch_info_from_labels(
            label_folder, image_width, image_height)
        
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0))
        self.confidence_records_11 = []
        self.confidence_records_non_11 = []
        self.best_non_target_patch = None
        self.best_non_target_conf_sum = -float('inf')
        self.best_epoch = 0
        self.utils = PatchUtils(self.device)

    def _load_patch_info_from_labels(self, label_folder, image_width, image_height):
        patch_sizes = []
        patch_locations = []

        label_files = glob.glob(os.path.join(label_folder, '*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    _, x_center, y_center, width, height = map(float, parts)
                    
                    patch_width = int(height * image_width)
                    patch_height = int(height * image_height)
                    patch_sizes.append(patch_width)
                    
                    x_top_left = int(x_center * image_width - width*image_width/2)
                    y_top_left = int(y_center * image_height - height*image_height/2)
                    
                    patch_locations.append((y_top_left, x_top_left))

        return patch_sizes, patch_locations

    # ... [其他方法保持不变，但移到这个类中] 