import json
import random
import numpy as np
import pandas as pd
from shutil import copy, rmtree
from pathlib import Path
from tqdm import tqdm
from PIL import Image

seed = 999
random.seed(seed)

src_dir = Path('~/ccpd_dataset/ccpd_base/').expanduser().resolve()
dst_dir = Path('./ccpd5000/')
train_dir = dst_dir / 'train'
valid_dir = dst_dir / 'valid'
test_dir = dst_dir / 'test'

if dst_dir.exists():
    rmtree(str(dst_dir))
for dir_ in [dst_dir, train_dir, valid_dir, test_dir]:
    dir_.mkdir(parents=True)

img_paths = list(src_dir.glob('**/*.jpg'))
img_paths = list(random.sample(img_paths, 6000))

random.shuffle(img_paths)
train_img_paths = img_paths[:4000]
valid_img_paths = img_paths[4000:5000]
test_img_paths = img_paths[5000:]

for img_path in tqdm(train_img_paths, desc='Train'):
    copy(img_path, str(train_dir))
for img_path in tqdm(valid_img_paths, desc='Valid'):
    copy(img_path, str(valid_dir))

anns = []
for i, img_path in enumerate(tqdm(test_img_paths, desc='Test')):
    copy(img_path, str(test_dir / f'{i:03d}.jpg'))

    img = Image.open(img_path)
    token = img_path.name.split('-')[3]
    token = token.replace('&', '_')
    kpt = [float(val) for val in token.split('_')]
    kpt = np.float32(kpt).reshape(4, 2)
    kpt = kpt / np.float32(img.size)
    kpt = kpt.flatten().tolist()

    anns.append([f'{i:03d}.jpg', *kpt])

columns = ['name', 'BR_x', 'BR_y', 'BL_x', 'BL_y', 'TL_x', 'TL_y', 'TR_x', 'TR_y']
anns = pd.DataFrame(anns)
anns.columns = columns
anns.to_csv('./test_true.csv', float_format='%.5f', index=False)

# tar zcvf ccpd5000.tar.gz ./ccpd5000/
