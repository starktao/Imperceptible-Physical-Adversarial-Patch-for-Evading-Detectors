import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2

class PatchUtils:
    def __init__(self, device):
        self.device = device

    def generate_patch(self, patch_size, type='gray'):
        if type == 'gray':
            return torch.full((3, patch_size, patch_size), 0.5, 
                            device=self.device, requires_grad=True)
        elif type == 'random':
            return torch.rand((3, patch_size, patch_size), 
                            device=self.device, requires_grad=True)

    def nps_calculator(self, patch):
        printability_array = torch.tensor([[0.2200, 0.2200, 0.2200], 
                                         [0.5010, 0.5010, 0.5010], 
                                         [0.8670, 0.8670, 0.8670]], 
                                        device=patch.device)
        diff = patch - printability_array.unsqueeze(2).unsqueeze(3)
        diff = torch.min(diff ** 2, dim=1)[0]
        diff = torch.sum(diff, dim=(1, 2))
        return torch.mean(diff)

    def total_variation(self, patch):
        tv_h = torch.mean(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]))
        tv_w = torch.mean(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]))
        return tv_h + tv_w

    def save_image(self, tensor, path, text=None):
        tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype('uint8')
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
        if text:
            cv2.putText(tensor, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(path, tensor) 