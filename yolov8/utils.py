import torch
import cv2
import numpy as np

def save_image(tensor, path, text=None):
    """保存图像的工具函数"""
    tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype('uint8')
    tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    if text:
        cv2.putText(tensor, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, tensor)

def nps_calculator(patch):
    """计算非打印性分数"""
    printability_array = torch.tensor(
        [[0.2200, 0.2200, 0.2200], 
         [0.5010, 0.5010, 0.5010], 
         [0.8670, 0.8670, 0.8670]], device=patch.device)
    diff = patch - printability_array.unsqueeze(2).unsqueeze(3)
    diff = torch.min(diff ** 2, dim=1)[0]
    diff = torch.sum(diff, dim=(1, 2))
    return torch.mean(diff)

def total_variation(patch):
    """计算总变差"""
    tv_h = torch.mean(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]))
    tv_w = torch.mean(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]))
    return tv_h + tv_w 