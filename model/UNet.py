import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers_per_block=2, features_start=32, bottle_neck_channels=None):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.features_start = features_start
        self.bottle_neck_channels = bottle_neck_channels or features_start

        # Down-sampling
        self.down1 = self._block(in_channels, features_start, layers_per_block)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = self._block(features_start, features_start * 2, layers_per_block)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = self._block(features_start * 2, features_start * 4, layers_per_block)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = self._block(features_start * 4, features_start * 8, layers_per_block)

        # Bottleneck
        self.bottleneck = self._block(features_start * 8, self.bottle_neck_channels, layers_per_block)

        # Up-sampling
        self.up4 = self._up_block(self.bottle_neck_channels, features_start * 8, layers_per_block)
        self.up3 = self._up_block(features_start * 8, features_start * 4, layers_per_block)
        self.up2 = self._up_block(features_start * 4, features_start * 2, layers_per_block)
        self.up1 = self._up_block(features_start * 2, features_start, layers_per_block)

        # Output layer
        self.out = nn.Conv2d(features_start, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels, layers, kernel_size=3, stride=1, padding=1):
        layers_list = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers_list += [nn.ReLU() for _ in range(layers - 1)]
        layers_list.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        layers_list.append(nn.ReLU())
        return nn.Sequential(*layers_list)

    def _up_block(self, in_channels, out_channels, layers, kernel_size=3, stride=1, padding=1):
        layers_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)]
        layers_list.append(nn.ReLU())
        layers_list += [self._block(out_channels, out_channels, layers, kernel_size, stride, padding)]
        return nn.Sequential(*layers_list)

    def forward(self, x):
        # Down-sampling
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x2 = self.down2(x2)
        x3 = self.pool2(x2)
        x3 = self.down3(x3)
        x4 = self.pool3(x3)
        x4 = self.down4(x4)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Up-sampling
        x6 = self.up4(x5 + x4)
        x7 = self.up3(x6 + x3)
        x8 = self.up2(x7 + x2)
        x9 = self.up1(x8 + x1)

        # Output layer
        out = self.out(x9)
        return out

# Example usage:
# Create a UNet model with 3 input channels and 3 output channels
unet_model = UNet(in_channels=3, out_channels=3)
