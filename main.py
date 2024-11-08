from model.Block1 import *
from torch import optim
# from model.NosePredictor import GaussianDiffusion
from model.UNet import Unet
import torch

DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.005
T = 1000
image_size = 32
channels = 1
IF_SMALL_BATCHES = True
dim_mults = (1, 2, 4,)

dataset_hsi, dataset_ndsm, dataset_rgb = GenerateDatasets(DATA_ROOT)
data_loader_hsi_train, data_loader_hsi_test = SpliteDataset(dataset_hsi, BATCH_SIZE, 0.8)
data_loader_ndsm_train, data_loader_ndsm_test = SpliteDataset(dataset_ndsm, BATCH_SIZE, 0.8)
data_loader_rgb_train, data_loader_rgb_test = SpliteDataset(dataset_rgb, BATCH_SIZE, 0.8)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
encoder_rgb = GenerateEncoders(3)

denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)
denoise_model = denoise_model.to(CUDA0)


classifier = Classifier().to(CUDA0)
criterion = CosineSimilarityLoss().to(CUDA0)
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# Train(data_loader_hsi_train, encoder_hsi, denoise_model, classifier, T, criterion, optimizer, CLS_EPOCH)
# Train(data_loader_ndsm_train, encoder_ndsm, denoise_model, classifier, T, criterion, optimizer, CLS_EPOCH)
Train(data_loader_rgb_train, encoder_rgb, denoise_model, classifier, T, criterion, optimizer, CLS_EPOCH)
# Test(data_loader_rgb_test, encoder_rgb, classifier)


