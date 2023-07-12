import os
import cv2
import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from PIL import Image

class ImagePairs(Dataset):
    def __init__(self,data_file,preprocess=lambda x:x,masked=False):
        self.dat = np.load(data_file)
        self.preprocess = preprocess
        self.masked = masked

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        im1_path,im2_path,label =  self.dat[idx]
        im1 = Image.open(im1_path)
        im2 = Image.open(im2_path)
        if self.masked:
            im1_mask = Image.open(im1_path.replace('outputs','output_masks'))
            im2_mask = Image.open(im2_path.replace('outputs','output_masks'))
            masked1 = np.array(im1) * (np.array(im1_mask)/255).astype(bool)
            masked2 = np.array(im2) * (np.array(im2_mask)/255).astype(bool)
            im1 = Image.fromarray(masked1)
            im2 = Image.fromarray(masked2)
        im1,im2 = [self.preprocess(x) for x in (im1,im2)]
        return im1,im2,label == "True"


def pad_images(image1, image2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Determine the target height and width
    target_height = max(height1, height2)
    target_width = max(width1, width2)

    # Compute the padding values
    pad_height1 = target_height - height1
    pad_width1 = target_width - width1
    pad_height2 = target_height - height2
    pad_width2 = target_width - width2

    # Pad the images with zeros
    padded_image1 = np.pad(image1, ((0, pad_height1), (0, pad_width1), (0, 0)), mode='constant')
    padded_image2 = np.pad(image2, ((0, pad_height2), (0, pad_width2), (0, 0)), mode='constant')

    return padded_image1, padded_image2

if __name__ == '__main__':
    dataset = ImagePairs("reid/data/fremont.npy")
    # randomly sample 50 elements from the dataset
    idx = np.random.choice(len(dataset), 50, replace=False)
    # create a grid
    # els = [ dataset[e][:2] for e in idx]
    from util.pyutil import write_images, flatten_images
    for i,idx in enumerate(idx):
        i1,i2,label = dataset[idx]
        ims = pad_images(i1,i2)
        ims = flatten_images(np.array(ims))
        # top left corner like 50 pixels wide
        cv2.putText(ims, str(int(label)), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=2)
        write_images(f'vis/fremont_dataset/{i}.png',np.array(ims))
