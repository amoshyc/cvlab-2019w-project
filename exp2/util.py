import math
import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import util
from skimage.transform import ProjectiveTransform, warp

def draw_kpts(img, kpts, c='red', r=1.0):
    '''Draw keypoints on image.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
        c: (PIL.Color) color of keypoints, default to 'red'
        r: (float) radius of keypoints, default to 1.0
    Return:
        img: (PIL.Image) modified image
    '''
    draw = ImageDraw.Draw(img)
    kpts = kpts.view(4, 2)
    kpts = kpts * torch.FloatTensor(img.size)
    kpts = kpts.numpy().tolist()
    for (x, y) in kpts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


def draw_plate(img, kpts):
    '''Perspective tranform and draw the plate indicated by kpts to a 96x30 rectangle.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
    Return:
        img: (PIL.Image) modified image
    Reference: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_geometric.html
    '''
    src = np.float32([[96, 30], [0, 30], [0, 0], [96, 0]])
    dst = kpts.view(4, 2).numpy()
    dst = dst * np.float32(img.size)

    transform = ProjectiveTransform()
    transform.estimate(src, dst)
    with warnings.catch_warnings(): # surpress skimage warning
        warnings.simplefilter("ignore")
        warped = warp(np.array(img), transform, output_shape=(30, 96))
        warped = util.img_as_ubyte(warped)
    plate = Image.fromarray(warped)
    img.paste(plate)
    return img


def gaussian2d(mu, sigma, shape):
    (r, c), (sr, sc), (H, W) = mu, sigma, shape
    pi = torch.tensor(math.pi)
    rr = torch.arange(r - 3 * sr, r + 3 * sr + 1).float()
    cc = torch.arange(c - 3 * sc, c + 3 * sc + 1).float()
    rr = rr[(rr >= 0) & (rr < H)]
    cc = cc[(cc >= 0) & (cc < W)]
    gr = torch.exp(-0.5 * ((rr - r) / sr)**2) / (torch.sqrt(2 * pi) * sr)
    gc = torch.exp(-0.5 * ((cc - c) / sc)**2) / (torch.sqrt(2 * pi) * sc)
    g = torch.ger(gr, gc).view(-1)
    R, C = len(rr), len(cc)
    rr = rr.long().contiguous().view(R, 1)
    cc = cc.long().contiguous().view(1, C)
    rr = rr.expand(R, C).contiguous().view(-1)
    cc = cc.expand(R, C).contiguous().view(-1)
    return rr, cc, g


def peek2d(lbl):
    _, H, W = lbl.size()
    lbl = lbl.view(4, -1) # [4, H * W]
    loc = lbl.argmax(dim=1, keepdim=True) # [4, 1]
    rr, cc = loc / W, loc % W # [4, 1], [4, 1]
    kpt = torch.cat((rr, cc), dim=1) # [4, 2]
    kpt = kpt.float() / torch.FloatTensor([H, W])
    kpt = kpt[:, [1, 0]] # rc -> xy
    return kpt


if __name__ == '__main__':
    img = torch.zeros(100, 100).float()
    rr, cc, g = gaussian2d([50, 50], [3, 3], shape=img.size())
    img[rr, cc] = g / g.max()

    from torchvision.transforms import functional as tf
    img = tf.to_pil_image(img.unsqueeze(dim=0))
    img.save('./out.png')
