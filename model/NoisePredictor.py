import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from tqdm import tqdm
from utils.util import EMA

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianDiffusion(nn.Module):
    def __init__(self, model, x_start, betas,
                 ema_decay=0.9999, ema_start=5000, ema_update_stride=1):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.x_start = x_start
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
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_prod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_stride == 0:
            if self.step < self.ema_start:
                pass
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    def diffuse(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def cosine_annealing_schedule(length_T, initial_beta, final=0.001):
    beta_schedual = initial_beta * (final / initial_beta) ** (0.5 * np.cos(np.pi * np.arange(length_T) / length_T))
    beta_schedual = torch.from_numpy(beta_schedual)
    return beta_schedual


def extract(a, t, x_shape):
    batch_size = x_shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


T = 1000
initial = 0.1

beta_array = cosine_annealing_schedule(T, initial).to(CUDA0)
alphas = 1.0 - beta_array
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)
posterior_variance = beta_array * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def q_sample(x_start, t, noise):
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    x_t = x_t.float()
    return x_t


def generate_single_X_0_hat(t_i, shape, noise_hat, alpha_t, beta_t,
                            sqrt_one_minus_alphas_cumprod_t_1,
                            sqrt_one_minus_alphas_cumprod_t,
                            sqrt_recip_alphas_t):
    X_t = torch.randn(shape[1:]).to(CUDA0)
    for j in range(t_i, 0, -1):
        if j == 0:
            z = torch.zeros(shape[1:]).to(CUDA0)
        else:
            z = torch.randn(shape[1:]).to(CUDA0)
        mu = sqrt_recip_alphas_t * (X_t - (1.0 - alpha_t) // sqrt_one_minus_alphas_cumprod_t * noise_hat)
        sigma = torch.sqrt(beta_t) * sqrt_one_minus_alphas_cumprod_t_1 / sqrt_one_minus_alphas_cumprod_t
        X_t_1 = mu + sigma * z
        X_t = X_t_1
    return X_t


def generate(shape, noise_hat, t):
    batch_size = shape[0]
    X_0s = torch.tensor([]).to(CUDA0)
    """shape = (32, 1, 32, 32)"""
    X_t = torch.randn(shape)
    alpha_t = extract(alphas, t, shape)
    beta_t = extract(beta_array, t, shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, shape
    )
    sqrt_one_minus_alphas_cumprod_t_1 = extract(
        sqrt_one_minus_alphas_cumprod, t-1, shape
    )
    for i in range(batch_size):
        X_0 = generate_single_X_0_hat(t[i], shape, noise_hat[i], alpha_t[i], beta_t[i],
                                      sqrt_one_minus_alphas_cumprod_t_1[i],
                                      sqrt_one_minus_alphas_cumprod_t[i],
                                      sqrt_recip_alphas[i])
        X_0 = torch.unsqueeze(X_0, dim=0)
        X_0s = torch.concatenate((X_0s, X_0))
    return X_0s
