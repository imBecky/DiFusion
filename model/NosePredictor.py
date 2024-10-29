import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 定义UNet的层，这里只是一个示例
        self.down1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def forward(self, x, t=None, encoder_hidden_states=None, cross_attention_kwargs=None):
        # 定义前向传播过程
        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(x1))
        x3 = F.relu(self.up1(x2))
        x4 = torch.sigmoid(self.up2(x3))
        return x4


def predict_noise(unet, input_data, t, encoder_hidden_states=None):
    # 使用UNet预测噪声
    noise_pred = unet(input_data, t, encoder_hidden_states)
    return noise_pred
