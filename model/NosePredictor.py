import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, model):
        super(GaussianDiffusion, self).__init__()
        self.model = model
