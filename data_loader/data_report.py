import torch

ROOT = '../data/tensor/'
hsi = torch.load(ROOT+'hsi.pth', weights_only=False)
ndsm = torch.load(ROOT+'ndsm.pth', weights_only=False)
rgb = torch.load(ROOT+'rgb.pth', weights_only=False)
gt = torch.load(ROOT+'gt.pth', weights_only=False)
print(f'hsi shape:{hsi.shape}')
print(f'ndsm shape:{ndsm.shape}')
print(f'rgb shape:{rgb.shape}')
print(f'gt shape:{gt.shape}')
