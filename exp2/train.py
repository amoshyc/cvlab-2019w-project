import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib as mpl
mpl.use('svg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import functional as tf

# For reproducibility
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import util
from dataset import CCPD5000
from model import CCPDModel

dataset = CCPD5000('./data/anns.json')
pivot = len(dataset) * 4 // 5
train_set = Subset(dataset, range(pivot))
valid_set = Subset(dataset, range(pivot, len(dataset)))
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 32)),
    Subset(valid_set, random.sample(range(len(valid_set)), 32)),
])
train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=1)
visul_loader = DataLoader(visul_set, 32, shuffle=False, num_workers=1)

device = 'cuda'
model = CCPDModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)

def train(pbar):
    model.train()
    losses = []
    for img_b, lbl_b in iter(train_loader):
        img_b = img_b.to(device)
        lbl_b = lbl_b.to(device)

        optimizer.zero_grad()
        pred_b = model(img_b)
        loss = criterion(pred_b, lbl_b)
        loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        losses.append(loss)
        pbar.set_postfix(loss=f'{loss:.5f}')
        pbar.update(img_b.size(0))
    
    avg_loss = sum(losses)/len(losses)
    pbar.set_postfix(avg_loss=f'{avg_loss:.5f}')
    return avg_loss


def valid(pbar):
    model.eval()
    losses = []
    for img_b, lbl_b in iter(valid_loader):
        img_b = img_b.to(device)
        lbl_b = lbl_b.to(device)
        pred_b = model(img_b)

        loss = criterion(pred_b, lbl_b)
        loss = loss.detach().item()
        losses.append(loss)

        pbar.set_postfix(loss=f'{loss:.5f}')
        pbar.update(img_b.size(0))
    
    avg_loss = sum(losses)/len(losses)
    pbar.set_postfix(avg_loss=f'{avg_loss:.5f}')
    return avg_loss


def visul(pbar):
    model.eval()
    epoch_dir = log_dir / f'{epoch:03d}'
    epoch_dir.mkdir()
    for img_b, lbl_b in iter(visul_loader):
        pred_b = model(img_b.to(device)).cpu()

        for img, pred_lbl, true_lbl in zip(img_b, pred_b, lbl_b):
            img = tf.to_pil_image(img)
            true_kpt = util.peek2d(true_lbl)
            pred_kpt = util.peek2d(pred_lbl)

            vis = util.draw_plate(img, pred_kpt)
            vis = util.draw_kpts(vis, true_kpt, c='orange')
            vis = util.draw_kpts(vis, pred_kpt, c='red')
            vis.save(epoch_dir / f'{pbar.n:03d}.vis1.jpg')
            
            lbls = torch.cat((true_lbl, pred_lbl), dim=0) # [8, H, W]
            lbls = lbls.unsqueeze(dim=1) # [8, 1, H, W]
            path = epoch_dir / f'{pbar.n:03d}.vis2.jpg'
            save_image(lbls, path, pad_value=1, nrow=4)

            pbar.update()


def log(epoch, train_loss, valid_loss):
    csv_path = log_dir / 'log.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
    }
    df = df.append(metrics, ignore_index=True)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(dpi=100, figsize=(10, 5))
    df[['train_loss', 'valid_loss']].plot(kind='line', ax=ax)
    fig.savefig(str(log_dir / 'loss.svg'))
    plt.close()

    if df['valid_loss'].idxmin() == epoch:
        torch.save(model, log_dir / 'model.pth')


for epoch in range(10):
    print('Epoch', epoch)
    with tqdm(total=len(train_set), desc='  Train') as pbar:
        train_loss = train(pbar)
    
    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid') as pbar:
            valid_loss = valid(pbar)
        with tqdm(total=len(visul_set), desc='  Visul') as pbar:
            visul(pbar)
        log(epoch, train_loss, valid_loss)