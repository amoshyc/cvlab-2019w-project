import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

class CCPD5000:
    def __init__(self, ann_path):
        ann_path = Path(ann_path)
        self.img_dir = ann_path.parent
        with ann_path.open() as f:
            self.anns = json.load(f)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]

        img_name = ann['img']
        img_path = self.img_dir / img_name
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = tf.to_tensor(img)
        
        kpt = torch.FloatTensor(ann['kpt'])
        kpt = kpt.view(-1) # [BR.x, BR.y, BL.x, BL.y, TL.x, TL.y, TR.x, TR.y]

        return img, kpt


if __name__ == '__main__':
    dataset = CCPD5000('./data/train/anns.json')
    print(len(dataset))
    
    img, kpt = dataset[-1]
    print(img.size())
    print(kpt.size())

    import util
    img = tf.to_pil_image(img)
    vis = util.draw_plate(img, kpt)
    vis = util.draw_kpts(img, kpt)
    vis.save('./check.png')

    # dataloader = DataLoader(dataset, 50)
    # for img_b, kpt_b in tqdm(iter(dataloader)):
    #     pass