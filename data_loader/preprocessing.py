import torch
import numpy as np
import torch.nn.functional as F
NDSM_PATH = '../data/tensor/ndsm.pth'

ndsm_tensor = torch.load(NDSM_PATH, weights_only=True)
# print(ndsm_tensor.shape)
indices = torch.where(ndsm_tensor > 400)
# print(len(indices[1]))
values_to_replace = ndsm_tensor[indices]
# print(values_to_replace)
left = ndsm_tensor[indices[0], indices[1]-1]
right = ndsm_tensor[indices[0], indices[1]+1]
top = ndsm_tensor[indices[0]-1, indices[1]]
bottom = ndsm_tensor[indices[0]+1, indices[1]]
avg = (left + right + top + bottom) / 4
print(avg)
ndsm_tensor[indices] = avg
print(ndsm_tensor[indices])
