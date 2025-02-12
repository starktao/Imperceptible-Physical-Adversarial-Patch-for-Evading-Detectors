import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from ultralytics.nn.autobackend import AutoBackend
import numpy as np
import os
import glob
from utils import save_image, nps_calculator, total_variation

class PatchTrainer:
    def __init__(self, mode, label_folder, image_width, image_height):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoBackend(weights="yolov8n.pt", device=self.device)
        self.model.to(self.device)

        self.patch_sizes, self.patch_locations = self.load_patch_info_from_labels(
            label_folder, image_width, image_height)

        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0))
        self.confidence_records_11 = []
        self.confidence_records_non_11 = []
        self.best_non_target_patch = None
        self.best_non_target_conf_sum = -float('inf')
        self.best_epoch = 0

    def load_patch_info_from_labels(self, label_folder, image_width, image_height):
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

    def generate_patch(self, type='gray', image_path="p.png"):
        if type == 'gray':
            adv_patch_cpu = torch.full((3, max(self.patch_sizes), max(self.patch_sizes)), 
                                     0.5, device=self.device, requires_grad=True)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, max(self.patch_sizes), max(self.patch_sizes)), 
                                     device=self.device, requires_grad=True)
        elif type == 'natural':
            if image_path is None:
                raise ValueError("Image path must be provided for 'natural' type")
            
            natural_image = Image.open(image_path)
            if natural_image.mode != 'RGB':
                natural_image = natural_image.convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((max(self.patch_sizes), max(self.patch_sizes))),
                transforms.ToTensor(),
            ])
            adv_patch_cpu = transform(natural_image).to(self.device)
            adv_patch_cpu.requires_grad = True

        return adv_patch_cpu

    def preprocess_image(self, image_path, img_size=640):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.type(torch.float32).to(self.device)

    def apply_patch(self, image_tensor, patch, img_index, epoch):
        y, x = self.patch_locations[img_index]
        patch_size = int(self.patch_sizes[img_index])
        _, _, H, W = image_tensor.shape

        if epoch < 6000:
            delta_x = delta_y = patch_size_delta = 0
        elif 6000 <= epoch < 12000:
            delta_x = delta_y = random.randint(-1, 1)
            patch_size_delta = random.randint(-1, 1)
        else:
            delta_x = delta_y = random.randint(-1, 1)
            patch_size_delta = random.randint(-1, 1)

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

        return torch.where(mask == 1, padded_patch, image_tensor)

    def detect_and_get_confidence(self, image_tensor, target_class=11):
        pred = self.model(image_tensor)
        if isinstance(pred, list):
            pred = pred[0]

        num_classes = 80
        class_conf = pred[:, 4:4 + num_classes]
        prediction = class_conf.transpose(-1, -2)
        max_cls_conf = prediction.amax(1)[0]
        
        top_confs = []
        for k in range(2, 6):
            top_k_conf, _ = torch.topk(prediction, k, dim=1)
            top_confs.append(top_k_conf[:, k-1][0])

        cls_sum = sum(top_confs) / len(top_confs)

        return cls_sum, max_cls_conf

    def color_similarity_loss(self, patch, img_tensor, img_index):
        y, x = self.patch_locations[img_index]
        patch_size = self.patch_sizes[img_index]
        img_patch = img_tensor[:, :, y:y + patch_size, x:x + patch_size]

        patch_resized = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), 
                                    mode='bilinear', align_corners=True)

        loss = F.mse_loss(patch_resized, img_patch)
        similarity_percentage = (1 - (loss / torch.tensor(1.0, device=self.device))) * 100

        return loss, similarity_percentage.item()

    def simulate_perspective_tensor(self, image_tensor, angle=45):
        _, _, H, W = image_tensor.shape
        angle_rad = angle * (np.pi / 180.0)

        src_points = [[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]]
        if angle > 0:
            dst_points = [
                [0, 0],
                [W - 1, H * np.tan(-angle_rad)],
                [0, H - 1],
                [W - 1, H - 1 - H * np.tan(-angle_rad)]
            ]
        else:
            dst_points = [
                [0, H * np.tan(angle_rad)],
                [W - 1, 0],
                [0, H - 1 - H * np.tan(angle_rad)],
                [W - 1, H - 1]
            ]

        return TF.perspective(image_tensor, startpoints=src_points, endpoints=dst_points)

    def train(self, img_tensor_list, n_epochs=3000):
        adv_patch_cpu = self.generate_patch("gray")
        optimizer = torch.optim.Adam([adv_patch_cpu], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = 0
            color_loss = 0
            color_loss_num = 10 if epoch >= 4000 else (5 if epoch >= 1200 else 0)

            images_to_train = img_tensor_list if epoch >= 1200 else [img_tensor_list[0]]
            non_target_conf_sum = conf_11_sum = 0

            for img_index, img_tensor in enumerate(images_to_train):
                patched_image = self.apply_patch(img_tensor, adv_patch_cpu, img_index, epoch)

                if epoch >= 5000:
                    angle = random.uniform(-5, 5)
                    patched_image = self.simulate_perspective_tensor(patched_image, angle)

                cls_sum, max_cls_conf = self.detect_and_get_confidence(patched_image)
                loss += cls_sum[11]

                max_value, max_index = torch.max(max_cls_conf, dim=0)
                if max_index == 11:
                    conf_11_sum += max_value.item()
                else:
                    non_target_conf_sum += max_value.item()

                if epoch % 200 == 0:
                    save_image(patched_image, 
                             f"saved_img530/patch_epoch_{epoch}_img_{img_index + 1}.jpg",
                             f"Class: {max_index}, Conf: {max_value:.2f}")

            total_loss = loss + total_variation(adv_patch_cpu) * 2.5
            if color_loss_num > 0:
                color_loss, _ = self.color_similarity_loss(adv_patch_cpu, img_tensor, img_index)
                total_loss += color_loss * color_loss_num

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            adv_patch_cpu.data.clamp_(0, 1)

            if epoch % 200 == 0:
                save_image(adv_patch_cpu, f"saved_patches530/adv_patch_epoch_{epoch}.jpg")
                if epoch >= 2500 and non_target_conf_sum > self.best_non_target_conf_sum:
                    self.best_non_target_conf_sum = non_target_conf_sum
                    self.best_non_target_patch = adv_patch_cpu.clone()
                    self.best_epoch = epoch

            if epoch % 30 == 0:
                self.confidence_records_11.append(conf_11_sum if epoch < 600 else conf_11_sum / 7)
                self.confidence_records_non_11.append(non_target_conf_sum if epoch < 600 else non_target_conf_sum / 7)

        print("Training completed.")
        save_image(adv_patch_cpu, "final_patch.jpg")
        if self.best_non_target_patch is not None:
            save_image(self.best_non_target_patch, "best_non_target_patch.jpg")
            print(f"Best epoch: {self.best_epoch}")

        self.plot_confidence_records()

    def plot_confidence_records(self):
        epochs = range(0, 3000, 30)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.confidence_records_11, label="Class 11 Confidence", color='blue')
        plt.plot(epochs, self.confidence_records_non_11, label="Non-Class 11 Confidence", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Confidence")
        plt.title("Confidence vs. Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig("confidence_vs_epochs2.jpg")
        plt.close() 