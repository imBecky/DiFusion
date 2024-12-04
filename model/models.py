import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from utils.util import extract, Reshape
from utils.params import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的网络结构
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播
        x = x.float()
        output = self.model(x)
        return output.view(-1, 1)


class Classifier(nn.Module):
    """TODO: deepen the model"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 16 * 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16 * 16),
            nn.Dropout(0.5),
            nn.Linear(16 * 16, 32 * 32),
            nn.ReLU(inplace=True),
            Reshape((32, 32))
        )

    def forward(self, x):
        x = self.model(x)
        return x


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_predictor_hsi, noise_predictor_ndsm, noise_predictor_rgb,
                 discriminator, classifier,
                 noise_predictor_criterion, generate_criterion, discriminator_criterion, classifier_criterion,
                 noise_predictor_optimizer_hsi, noise_predictor_optimizer_ndsm, noise_predictor_optimizer_rgb,
                 discriminator_optimizer, classifier_optimizer,
                 betas, ema_decay=0.9999, ema_start=5000, ema_update_stride=1):
        super(GaussianDiffusion, self).__init__()
        self.noise_predictor_hsi = noise_predictor_hsi
        self.noise_predictor_ndsm = noise_predictor_ndsm
        self.noise_predictor_rgb = noise_predictor_rgb
        self.discriminator = discriminator
        self.classifier = classifier
        self.noise_predictor_criterion = noise_predictor_criterion
        self.generate_criterion = generate_criterion
        self.discriminator_criterion = discriminator_criterion
        self.classifier_criterion = classifier_criterion
        self.noise_predictor_optimizer_hsi = noise_predictor_optimizer_hsi
        self.noise_predictor_optimizer_ndsm = noise_predictor_optimizer_ndsm
        self.noise_predictor_optimizer_rgb = noise_predictor_optimizer_rgb
        self.discriminator_optimizer = discriminator_optimizer
        self.classifier_optimizer = classifier_optimizer
        # self.ema_model = copy.deepcopy(noise_predictor)
        # self.ema = EMA(ema_decay)
        # self.ema_decay = ema_decay
        # self.ema_start = ema_start
        # self.ema_update_stride = ema_update_stride
        self.step = 0

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_prod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_prod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)

    # def update_ema(self):
    #     self.step += 1
    #     if self.step % self.ema_update_stride == 0:
    #         if self.step < self.ema_start:
    #             pass
    #         else:
    #             self.ema.update_model_average(self.ema_model, self.noise_predictor)

    def diffuse(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        x_t = x_t.float()
        return x_t

    def generate_single(self, t_i, shape, noise_hat, alpha_t, beta_t,
                        sqrt_one_minus_alphas_cumprod_t_1,
                        sqrt_one_minus_alphas_cumprod_t,
                        sqrt_recip_alphas_t
                        ):
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

    def generate(self, shape, epsilon, t):
        batch_size = shape[0]
        X_0_hats = torch.tensor([]).to(CUDA0)
        alpha_t = extract(self.alphas, t, shape)
        beta_t = extract(self.betas, t, shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, shape
        )
        sqrt_one_minus_alphas_cumprod_t_1 = extract(
            self.sqrt_one_minus_alphas_cumprod, t - 1, shape
        )
        for i in range(batch_size):
            X_0 = self.generate_single(t[i], shape, epsilon[i], alpha_t[i], beta_t[i],
                                       sqrt_one_minus_alphas_cumprod_t_1[i],
                                       sqrt_one_minus_alphas_cumprod_t[i],
                                       self.sqrt_recip_alphas[i])
            X_0 = torch.unsqueeze(X_0, dim=0)
            X_0_hats = torch.concatenate((X_0_hats, X_0))
        return X_0_hats


class MyEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MyEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = self.gen_encoder()
        self.reshape = Reshape((1, 32, 32))

    def gen_encoder(self):
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        encoder.conv1 = torch.nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        encoder.fc = torch.nn.Linear(2048, 1024, bias=True)
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.reshape(x)
        return x


def GenerateEncoders(option=0):
    if option == 0:
        print('No encoder generated!')
    elif option == 1:
        encoder1 = MyEncoder(48)
        encoder1 = encoder1.to(CUDA0)
        return encoder1
    elif option == 2:
        encoder2 = MyEncoder(1)
        encoder2 = encoder2.to(CUDA0)
        return encoder2
    elif option == 3:
        encoder3 = MyEncoder(3)
        encoder3 = encoder3.to(CUDA0)
        return encoder3
