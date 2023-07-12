from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, ToPILImage
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from reid.utils import ResizeLargest, PadToSize

im = np.ones((200,100))
transform_aspect = Compose([
    ResizeLargest(224),
    PadToSize(224,equal_padding=True),
    ToTensor(),
    # CenterCrop(224)
])

from util.pyutil import write_images
transform_aspect(Image.fromarray(im)).shape
write_images('vis/test.png',transform_aspect(Image.fromarray(im)))


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

