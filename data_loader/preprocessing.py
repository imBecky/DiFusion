import torch
import numpy as np
import torch.nn.functional as F
HSI_PATH = '../data/tensor/hsi.pth'

hsi_tensor = torch.load(HSI_PATH).float()
hsi_tensor = np.transpose(hsi_tensor, (0, 2, 1))
print(hsi_tensor.shape)
hsi_tensor = hsi_tensor.unsqueeze(0).transpose(1, 3).transpose(2, 3)
new_height = 1202
new_width = 4768
upsampled_tensor = F.interpolate(hsi_tensor, scale_factor=2, mode='bilinear', align_corners=False)
print(upsampled_tensor.shape)

