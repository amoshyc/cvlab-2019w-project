import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, (3, 3), padding=1, stride=2)
        self.conv2 = nn.Conv2d(cout, cout, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act1(self.bn2(self.conv2(x)))
        return x


class DeConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, (2, 2), stride=2)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.bn(self.deconv(x)))
        return x


class CCPDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            DeConvBlock(64, 32),
            DeConvBlock(32, 16),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.Conv2d(16, 4, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class CCPDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_lbl_b, true_lbl_b):
        N, _, H, W = pred_lbl_b.size()
        device = pred_lbl_b.device

        loss_b = F.binary_cross_entropy(pred_lbl_b, true_lbl_b, reduction='none')
        weight_b = torch.full((N, 4, H, W), 1.0, device=device)
        weight_b[true_lbl_b > 1e-6] = 64.0
        loss = (loss_b * weight_b).sum() / weight_b.sum()

        return loss


if __name__ == '__main__':
    device = 'cuda'
    model = CCPDModel().to(device)
    img_b = torch.rand(16, 3, 192, 320).to(device)
    out_b = model(img_b)
    print(out_b.size())
