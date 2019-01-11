import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree


def process(src_dir, dst_dir, img_size):
    img_paths = list(src_dir.glob('*.jpg'))
    img_paths = sorted(img_paths)
    dst_dir.mkdir(exist_ok=True, parents=True)

    anns = []
    desc = f'{src_dir} -> {dst_dir}'
    for img_path in tqdm(img_paths, desc=desc):
        dst_path = dst_dir / img_path.name
        
        img = Image.open(img_path)
        src_size = img.size # [w, h]
        img = img.convert('RGB')
        img = img.resize(img_size) # [w, h]
        img.save(dst_path)

        token = img_path.name.split('-')[3]
        token = token.replace('&', '_')
        kpt = [float(val) for val in token.split('_')]
        kpt = torch.tensor(kpt).view(4, 2)
        kpt = kpt / torch.FloatTensor(src_size)

        anns.append({
            'img': str(img_path.name),
            'kpt': kpt.numpy().tolist(),
        })

    with (dst_dir / 'anns.json').open('w') as f:
        json.dump(anns, f, indent=2, ensure_ascii=False)


img_size = (192, 320)
dataset_dir = Path('../raw/ccpd5000/')
target_dir = Path('./data/')
if target_dir.exists():
    rmtree(str(target_dir))

process(dataset_dir / 'train', target_dir / 'train', img_size)
process(dataset_dir / 'valid', target_dir / 'valid', img_size)