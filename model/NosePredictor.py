import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.util import EMA


class GaussianDiffusion(nn.Module):
    def __init__(self, model, ema_decay=0.9999, ema_start=5000, ema_update_stride=1):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.ema = EMA(self.ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_stride = ema_update_stride
        self.step = 0

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_stride == 0:
            if self.step < self.ema_start:
                pass
            else:
                self.ema.update_model_average(self.ema_model, self.model)

