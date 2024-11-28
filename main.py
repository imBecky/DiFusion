from model.Block1 import *
from torch import optim
from model.NoisePredictor import Discriminator, GaussianDiffusion, cosine_annealing_schedule
from model.UNet import Unet
import torch
import os
from params import *

dataset = GenerateDatasets(DATA_ROOT)
data_loader_train, data_hsi_test = SpliteDataset(dataset, BATCH_SIZE, 0.8)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
encoder_rgb = GenerateEncoders(3)
noise_predictor_hsi = Unet(dim=image_size, channels=feature_channels, dim_mults=dim_mults)
noise_predictor_ndsm = Unet(dim=image_size, channels=feature_channels, dim_mults=dim_mults)
noise_predictor_rgb = Unet(dim=image_size, channels=feature_channels, dim_mults=dim_mults)
beta_array = cosine_annealing_schedule(T, 0.1)
discriminator = Discriminator()
classifier = Classifier()
noise_predictor_criterion = F.smooth_l1_loss
generate_criterion = nn.MSELoss()
discriminator_criterion = CosineSimilarityLoss()
classifier_criterion = CosineSimilarityLoss()
noise_predictor_optimizer_hsi = optim.Adam(noise_predictor_hsi.parameters(), lr=LEARNING_RATE1)
noise_predictor_optimizer_ndsm = optim.Adam(noise_predictor_ndsm.parameters(), lr=LEARNING_RATE1)
noise_predictor_optimizer_rgb = optim.Adam(noise_predictor_rgb.parameters(), lr=LEARNING_RATE1)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE2)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE3)

for i, model in enumerate([encoder_rgb, encoder_hsi, encoder_ndsm,
                           noise_predictor_hsi, noise_predictor_ndsm, noise_predictor_rgb,
                           discriminator, classifier]):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'模型总参数数量: {num_params}')

GaussianDiffuser = GaussianDiffusion(encoder_hsi, encoder_ndsm, encoder_rgb,
                                     noise_predictor_hsi, noise_predictor_ndsm, noise_predictor_rgb,
                                     discriminator, classifier,
                                     noise_predictor_criterion,
                                     generate_criterion,
                                     discriminator_criterion,
                                     classifier_criterion,
                                     noise_predictor_optimizer_hsi,
                                     noise_predictor_optimizer_ndsm,
                                     noise_predictor_optimizer_rgb,
                                     discriminator_optimizer, classifier_optimizer,
                                     beta_array)
GaussianDiffuser = GaussianDiffuser.to(CUDA0)
Train(data_loader_train, GaussianDiffuser, CLS_EPOCH)  # stage1:encode, stage2:afterwards
# Test(data_loader_test, encoder_rgb, classifier)

