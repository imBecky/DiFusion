from model.Block1 import *
from torch import optim
from model.NoisePredictor import Discriminator, GaussianDiffusion, cosine_annealing_schedule
from model.UNet import Unet
import torch

DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.005
T = 1000
image_size = 32
channels = 1
IF_SMALL_BATCHES = True
dim_mults = (1, 2, 4,)


dataset = GenerateDatasets(DATA_ROOT)
data_loader_train, data_hsi_test = SpliteDataset(dataset, BATCH_SIZE, 0.8)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
encoder_rgb = GenerateEncoders(3)

noise_predictor = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)
noise_predictor = noise_predictor.to(CUDA0)
beta_array = cosine_annealing_schedule(T, 0.1).to(CUDA0)
discriminator = Discriminator().to(CUDA0)
classifier = Classifier().to(CUDA0)
criterion = CosineSimilarityLoss().to(CUDA0)
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
GaussianDiffuser = GaussianDiffusion(encoder_hsi, encoder_ndsm, encoder_rgb, noise_predictor,
                                     discriminator, classifier, beta_array)

Train(data_loader_train, GaussianDiffuser, classifier, T, criterion, optimizer, CLS_EPOCH)
# Test(data_loader_test, encoder_rgb, classifier)


