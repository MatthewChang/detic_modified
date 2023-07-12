from glob import glob
import util
from util.pyutil import write_images
import os
import torch
from pathlib import Path
files = glob("outputs/*/*/*/*.png",recursive=True)
dats = glob("outputs/*/*.torch",recursive=True)
sources = {}
for d in (dats):
    dat = torch.load(d)
    parts = Path(d).parts
    datid = '/'.join(parts[1:3])[:-6]
    sources[datid] = dat

from tqdm import tqdm
for f in tqdm(files):
    parts = Path(f).parts
    datid = '/'.join(parts[1:3])
    dat = sources[datid]
    imid = parts[-1][:-6]
    instance_id = int(parts[-1][-5])
    full_mask = dat[imid]['masks'][instance_id]
    bbox = dat[imid]['boxes'][instance_id]
    bbox = bbox.int()
    # select the mask based on bbox
    mask = full_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    mask_out = os.path.join('output_masks',*parts[1:])
    os.makedirs(os.path.dirname(mask_out),exist_ok=True)
    write_images(mask_out,mask.float().cpu())
