import random
from shutil import copy, rmtree
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

seed = 999
random.seed(seed)

src_dir = Path('~/ccpd_dataset/ccpd_base/').expanduser().resolve()
dst_dir = Path('./ccpd5000/')
train_dir = dst_dir / 'train'
valid_dir = dst_dir / 'valid'
if dst_dir.exists():
    rmtree(str(dst_dir))
    for dir_ in [dst_dir, train_dir, valid_dir]:
        dir_.mkdir(parents=True)

img_paths = list(src_dir.glob('**/*.jpg'))
img_paths = list(random.sample(img_paths, 5000))

random.shuffle(img_paths)
pivot = len(img_paths) * 4 // 5
train_img_paths = img_paths[:pivot]
valid_img_paths = img_paths[pivot:]

for img_path in tqdm(train_img_paths, desc='Train'):
    copy(img_path, str(train_dir))
for img_path in tqdm(valid_img_paths, desc='Valid'):
    copy(img_path, str(valid_dir))

# tar zcvf ccpd5000.tar.gz ./ccpd5000/
