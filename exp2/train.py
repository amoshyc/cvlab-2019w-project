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
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import functional as tf

# For reproducibility
# Set before loading model and dataset
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import util
from dataset import CCPD5000
from model import CCPDModel, CCPDLoss

train_set = CCPD5000('./data/train/anns.json')
valid_set = CCPD5000('./data/valid/anns.json')
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 32)),
    Subset(valid_set, random.sample(range(len(valid_set)), 32)),
])
train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=1)
visul_loader = DataLoader(visul_set, 32, shuffle=False, num_workers=1)

device = 'cuda'
model = CCPDModel().to(device)
criterion = CCPDLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)
history = {
    'train_bce': [],
    'valid_bce': [],
    'train_mse': [],
    'valid_mse': []
}

def train(pbar):
    model.train()
    bce_steps = []
    mse_steps = []

    for img_b, lbl_true_b, kpt_true_b in iter(train_loader):
        img_b = img_b.to(device)
        lbl_true_b = lbl_true_b.to(device)
        kpt_true_b = kpt_true_b.to(device)

        optimizer.zero_grad()
        lbl_pred_b = model(img_b)
        loss = criterion(lbl_pred_b, lbl_true_b)
        loss.backward()
        optimizer.step()

        bce = loss.detach().item()
        bce_steps.append(bce)
        kpt_pred_b = util.peek2d(lbl_pred_b.detach())
        mse = F.mse_loss(kpt_pred_b, kpt_true_b).item()
        mse_steps.append(mse)

        pbar.set_postfix(bce=bce, mse=mse)
        pbar.update(img_b.size(0))

    avg_bce = sum(bce_steps) / len(bce_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_bce=f'{avg_bce:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['train_bce'].append(avg_bce)
    history['train_mse'].append(avg_mse)


def valid(pbar):
    model.eval()
    bce_steps = []
    mse_steps = []

    for img_b, lbl_true_b, kpt_true_b in iter(valid_loader):
        img_b = img_b.to(device)
        lbl_true_b = lbl_true_b.to(device)
        kpt_true_b = kpt_true_b.to(device)

        lbl_pred_b = model(img_b)
        loss = criterion(lbl_pred_b, lbl_true_b)

        bce = loss.detach().item()
        bce_steps.append(bce)
        kpt_pred_b = util.peek2d(lbl_pred_b.detach())
        mse = F.mse_loss(kpt_pred_b, kpt_true_b).item()
        mse_steps.append(mse)

        pbar.set_postfix(bce=bce, mse=mse)
        pbar.update(img_b.size(0))

    avg_bce = sum(bce_steps) / len(bce_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_bce=f'{avg_bce:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['valid_bce'].append(avg_bce)
    history['valid_mse'].append(avg_mse)


def visul(pbar):
    model.eval()
    epoch_dir = log_dir / f'{epoch:03d}'
    epoch_dir.mkdir()
    for img_b, lbl_true_b, kpt_true_b in iter(visul_loader):
        lbl_pred_b = model(img_b.to(device)).cpu()
        kpt_pred_b = util.peek2d(lbl_pred_b)

        for i in range(img_b.size(0)):
            img = tf.to_pil_image(img_b[i])
            lbl_true = lbl_true_b[i]
            lbl_pred = lbl_pred_b[i]
            kpt_true = kpt_true_b[i]
            kpt_pred = kpt_pred_b[i]

            vis = util.draw_plate(img, kpt_pred)
            vis = util.draw_kpts(vis, kpt_true, c='orange')
            vis = util.draw_kpts(vis, kpt_pred, c='red')
            vis.save(epoch_dir / f'{pbar.n:03d}.vis1.jpg')

            lbls = torch.cat((lbl_true, lbl_pred), dim=0) # [8, H, W]
            lbls = lbls.unsqueeze(dim=1) # [8, 1, H, W]
            path = epoch_dir / f'{pbar.n:03d}.vis2.jpg'
            save_image(lbls, path, pad_value=1, nrow=4)

            pbar.update()


def log(epoch, train_loss, valid_loss):
    with (log_dir / 'metrics.json').open('w') as f:
        json.dump(history, f)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    ax[0].set_title('BCE')
    ax[0].plot(range(epoch + 1), history['train_bce'], label='Train')
    ax[0].plot(range(epoch + 1), history['valid_bce'], label='Valid')
    ax[0].legend()
    ax[1].set_title('MSE')
    ax[1].plot(range(epoch + 1), history['train_mse'], label='Train')
    ax[1].plot(range(epoch + 1), history['valid_mse'], label='Valid')
    ax[1].legend()
    fig.savefig(log_dir / 'metrics.jpg')
    plt.close()


for epoch in range(20):
    print('Epoch', epoch)
    with tqdm(total=len(train_set), desc='  Train') as pbar:
        train_loss = train(pbar)

    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid') as pbar:
            valid_loss = valid(pbar)
        with tqdm(total=len(visul_set), desc='  Visul') as pbar:
            visul(pbar)
        log(epoch, train_loss, valid_loss)
