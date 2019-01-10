import random
from shutil import copy, rmtree
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

src_dir = Path('~/ccpd_dataset/ccpd_base/').expanduser().resolve()
dst_dir = Path('./raw/ccpd5000/')
if dst_dir.exists():
    rmtree(str(dst_dir))
    dst_dir.mkdir(exist_ok=True, parents=True)

img_paths = list(src_dir.glob('**/*.jpg'))
img_paths = list(random.sample(img_paths, 5000))

for img_path in tqdm(img_paths):
    copy(img_path, str(dst_dir))