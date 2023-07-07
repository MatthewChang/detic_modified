from os.path import basename, splitext
from tqdm import tqdm
import imageio
from glob import glob
from util.pyutil import write_sequence
data_dir = "/private/home/matthewchang/data"
videos = glob(f"{data_dir}/*.mp4",recursive=True)
for video in tqdm(videos):
    mimread = imageio.get_reader(video,  'ffmpeg')
    fname = splitext(basename(video))[0]
    write_sequence(f'./vis/frames/{fname}',tqdm(mimread),pad_num=10,extension='jpg')

