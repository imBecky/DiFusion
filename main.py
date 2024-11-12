from model.Block1 import *
from torch import optim
from model.NoisePredictor import Discriminator, GaussianDiffusion, cosine_annealing_schedule
from model.UNet import Unet
import torch
from params import *


dataset = GenerateDatasets(DATA_ROOT)
data_loader_train, data_hsi_test = SpliteDataset(dataset, BATCH_SIZE, 0.8)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
encoder_rgb = GenerateEncoders(3)
noise_predictor = Unet(dim=image_size, channels=feature_channels, dim_mults=dim_mults)
beta_array = cosine_annealing_schedule(T, 0.1)
discriminator = Discriminator()
classifier = Classifier()
noise_predictor_criterion = F.smooth_l1_loss
discriminator_criterion = nn.BCELoss()
classifier_criterion = CosineSimilarityLoss()
noise_predictor_optimizer = optim.Adam(noise_predictor.parameters(), lr=LEARNING_RATE1)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE2)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE3)

GaussianDiffuser = GaussianDiffusion(encoder_hsi, encoder_ndsm, encoder_rgb,
                                     noise_predictor, discriminator, classifier,
                                     noise_predictor_criterion, discriminator_criterion, classifier_criterion,
                                     noise_predictor_optimizer, discriminator_optimizer, classifier_optimizer,
                                     beta_array)
GaussianDiffuser = GaussianDiffuser.to(CUDA0)
Train(data_loader_train, GaussianDiffuser, classifier, T, CLS_EPOCH)
# Test(data_loader_test, encoder_rgb, classifier)


