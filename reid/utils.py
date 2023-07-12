from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

class ResizeLargest(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        tensor_in = False
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
            tensor_in = True

        width, height = img.size
        aspect_ratio = float(width) / float(height)

        if width > height:
            new_width = self.size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.size
            new_width = int(new_height * aspect_ratio)

        img = img.resize((new_width, new_height), resample=Image.BILINEAR)
        if tensor_in:
            img = ToTensor()(img)
        return img


class PadToSize(object):
    def __init__(self, size, equal_padding=False):
        self.size = size
        self.equal_padding = equal_padding

    def __call__(self, img):
        tensor_in = False
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
            tensor_in = True

        width, height = img.size

        if width >= self.size and height >= self.size:
            return img

        pad_width = max(self.size - width, 0)
        pad_height = max(self.size - height, 0)

        if self.equal_padding:
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            padding = (pad_left, pad_top, pad_right, pad_bottom)  # (left, top, right, bottom)
        else:
            padding = (0, 0, pad_width, pad_height)  # (left, top, right, bottom)

        img = F.pad(img, padding, fill=0)
        if tensor_in:
            img = ToTensor()(img)
        return img

