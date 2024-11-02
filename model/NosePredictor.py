import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from utils.util import EMA


class GaussianDiffusion(nn.Module):
    def __init__(self, model, betas,
                 ema_decay=0.9999, ema_start=5000, ema_update_stride=1):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.betas = betas
        self.ema_model = copy.deepcopy(model)
        self.ema = EMA(self.ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_stride = ema_update_stride
        self.step = 0

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_prod = alphas_cumprod ** 0.5
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_prod', sqrt_alphas_prod)

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_stride == 0:
            if self.step < self.ema_start:
                pass
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    def add_noise(self, X0, t, noise):
        """TODO: get noise in the def rather than get it """
        sqrt_alpha_prod = self.sqrt_alpha_prod[:t]
        sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[:t]) ** 0.5
        X_t = sqrt_alpha_prod * X0 + sqrt_one_minus_alpha_prod * noise
        return X_t


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
