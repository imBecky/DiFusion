import torch

ROOT = '../data/tensor/'
# feature_hsi = torch.load(ROOT + 'features/' + 'hsi.pth', weights_only=False)
# feature_ndsm = torch.load(ROOT + 'features/' + 'ndsm.pth', weights_only=False)
# feature_rgb = torch.load(ROOT + 'features/' + 'rgb.pth', weights_only=False)
feature_gt = torch.load(ROOT + 'features/' + 'gt.pth', weights_only=False)
small_feature = feature_gt[:2048]
torch.save(small_feature, ROOT+'small/gt.pth')
