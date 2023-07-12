import torch
import numpy as np
import clip
from reid.datasets.image_pairs import ImagePairs
import argparse
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from util.pyutil import write_images
from superglue.matching import Matching
from superglue.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda

parser = argparse.ArgumentParser(description='')
parser.add_argument('--method',choices=['clip','superglue'],default='clip')
parser.add_argument('--mask',action='store_true')
parser.add_argument('--pre-process',choices=['default','paste_center'],default='default')
# parser.add_argument('--seed',default=0,type=int)
args = parser.parse_args()

# places an image in the center of a black image of size x size
def paste_center(size,im):
    h,w = im.shape[:2]
    size = max(size,h,w)
    out = np.zeros((size,size,3),dtype=np.uint8)
    out[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w] = im
    return out

class ClipEval():
    def __init__(self):
        self.model, self.rescale = clip.load("ViT-B/32")
        self.device = 'cuda'
        if args.pre_process == 'paste_center':
            def process(im):
                centered = paste_center(224,np.array(im))
                return self.rescale(Image.fromarray(centered))
            self.preprocess = process
        else:
            self.preprocess = self.rescale
    def __call__(self,i1,i2):
        with torch.no_grad():
            pi1,pi2 = [x.unsqueeze(0).to(self.device) for x in [i1,i2]]
            feats = [self.model.encode_image(x) for x in [pi1,pi2]]
            return torch.cosine_similarity(*feats).cpu().numpy()

class SuperpointEval():
    def __init__(self):
        self.device = 'cuda'
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(config).eval().to(self.device)
        # grayscale image
        if args.pre_process == 'paste_center':
            self.preprocess = Compose([Lambda(lambda x: paste_center(224,np.array(x))),ToTensor(),Lambda(lambda x: x[-1:])])
        else:
            self.preprocess = Compose([ToTensor(),Lambda(lambda x: x[-1:])])

    def __call__(self,f1,f2):
        # add batch dim
        min_size = min(f1.shape[1:] + f2.shape[1:])
        if min_size <= 7:
            return 0
        with torch.no_grad():
            f1,f2 = [x.unsqueeze(0).to(self.device) for x in [f1,f2]]
            keys = ['keypoints', 'scores', 'descriptors']
            last_data = self.matching.superpoint({'image': f1})
            last_data = {k+'0': last_data[k] for k in keys}
            last_data['image0'] = f1
            # frame_tensor = frame2tensor(f2, self.device)
            pred = self.matching({**last_data, 'image1': f2})
            kpts0 = last_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            valid = matches > -1
            # color = cm.jet(confidence[valid])
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            k_thresh = self.matching.superpoint.config['keypoint_threshold']
            m_thresh = self.matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]
            if(np.sum(valid) > 2):
                return np.sum(valid)/len(kpts0)
            else:
                return 0
        # out = make_matching_plot_fast(
            # f1, f2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            # path=None, show_keypoints=True, small_text=small_text)
        # return out,np.sum(valid)

if args.method == 'clip':
    model = ClipEval()
elif args.method == 'superglue':
    model = SuperpointEval()
else:
    raise NotImplementedError

# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
# from torchvision.transforms import InterpolationMode
# BICUBIC = InterpolationMode.BICUBIC
# def _transform(n_px):
    # return Compose([
        # # Lambda(lambda x: x.float()/255),
        # Resize(n_px, interpolation=BICUBIC),
        # CenterCrop(n_px),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # ])

# trans = _transform(model.model.visual.input_resolution)
# trans(i1[0])
dataset = ImagePairs("reid/data/fremont.npy",preprocess=model.preprocess,mask=args.mask)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
res = []
labels = []
for batch in tqdm(dataloader):
    i1,i2,label = batch
    res.append(model(i1[0],i2[0]))
    labels.append(label[0].item())
res = np.array(res)
labels = np.array(labels).astype(int)

# compute mean average precision using sklean
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
average_precision = average_precision_score(labels, res)

print('Average precision-recall score: {0:0.3f}'.format( average_precision))
print('baseline %.03f' % (labels.sum()/len(labels)))

display = PrecisionRecallDisplay.from_predictions(labels, res)
display.plot()
import matplotlib.pyplot as plt
plt.savefig(f'vis/pr_{args.method}.png')

# plt.clf()
# randoms = np.random.rand(*res.shape)
# display = PrecisionRecallDisplay.from_predictions(labels, randoms)
# display.plot()
# import matplotlib.pyplot as plt
# plt.savefig(f'vis/pr_random.png')
