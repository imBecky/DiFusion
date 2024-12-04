import torch

ROOT = '../data/tensor/'
hsi = torch.load(ROOT + 'hsi.pth', weights_only=False)
ndsm = torch.load(ROOT + 'ndsm.pth', weights_only=False)
rgb = torch.load(ROOT + 'rgb.pth', weights_only=False)
gt = torch.load(ROOT + 'gt.pth', weights_only=False)
print(f'hsi shape:{hsi.shape}')
print(f'ndsm shape:{ndsm.shape}')
print(f'rgb shape:{rgb.shape}')
print(f'gt shape:{gt.shape}')
feature_hsi = torch.load(ROOT + 'features/' + 'hsi.pth', weights_only=False)
feature_ndsm = torch.load(ROOT + 'features/' + 'ndsm.pth', weights_only=False)
feature_rgb = torch.load(ROOT + 'features/' + 'rgb.pth', weights_only=False)
feature_gt = torch.load(ROOT + 'features/' + 'gt.pth', weights_only=False)
print(f'feature_hsi shape:{feature_hsi.shape}')
print(f'feature_ndsm shape:{feature_ndsm.shape}')
print(f'feature_rgb shape:{feature_rgb.shape}')
print(f'feature_gt shape:{feature_gt.shape}')

