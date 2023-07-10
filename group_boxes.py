import clip
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
model, preprocess = clip.load("ViT-B/32")
device = "cuda" if torch.cuda.is_available() else "cpu"
im_files = glob("outputs/cup/2023-07-05-16-29-54/*.png",recursive=True)
# compute the clip features for each file
features = []
for im in tqdm(im_files):
    with torch.no_grad():
        img = preprocess(Image.open(im)).unsqueeze(0).to(device)
        feats = model.encode_image(img)
        features.append(feats[0].cpu().numpy())

# compute k-means clustering of clip features
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=8).fit(np.array(features))
print(kmeans.labels_)
print(kmeans.cluster_centers_)

# write out examles of each cluster
import shutil
import os
from itertools import groupby
labeled = list(zip(im_files,kmeans.labels_))
std = sorted(labeled,key=lambda x: x[1])
grouped = [(k,list(v)) for k,v in groupby(std,lambda x: x[1])]

# randomly sample 10 from each grouped
for cluster,files in grouped:
    os.makedirs(f"vis/cup/2023-07-05-16-29-54",exist_ok=True)
    # randomly sample 10 from files
    files = [ e[0] for e in files]
    files = np.random.choice(files,10)
    for i,im in enumerate(files):
        shutil.copy(im,f"vis/cup/2023-07-05-16-29-54/cluster_{cluster}_{i}.png")


target = model.encode_text(clip.tokenize(["a light blue cup"]).cuda())
# compute the distance between each image and the target
distances = []
for im in tqdm(im_files):
    with torch.no_grad():
        img = preprocess(Image.open(im)).unsqueeze(0).to(device)
        feats = model.encode_image(img)
        distances.append(torch.cosine_similarity(target,feats).cpu().numpy())

# find the top 10 images
dists = np.array(distances)[:,0]
dists.min()
dists.max()
top10 = np.argsort(dists)[::-1][:10]
dists[top10]
for i,im in enumerate(top10):
    shutil.copy(im_files[im],f"vis/cup/2023-07-05-16-29-54/top_{i}.png")

