import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import Reshape


class InBlock(nn.Module):
    def __init__(self, in_channel):
        super(InBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channel, 64*64*3),
            nn.BatchNorm1d(5),
            nn.Linear(64*64*3, 32*32*3),
            Reshape((3, 32, 32))
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvBlock(in_channel, out_channel)

    def forward(self, x2, x1):
        x2 = self.up(x2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inBlock = InBlock(in_channels)
        self.inc = ConvBlock((3, 32, 32), 32)
        self.down1 = DownSample(32, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)
        self.up3 = UpSample(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x0 = self.inBlock(x)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


