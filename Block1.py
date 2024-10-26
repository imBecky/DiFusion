import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from data_loader.dataset import HsiDataset

HSI_PATH = './data/tensor/hsi.pth'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 10
IF_SMALL_BATCHES = True

data_hsi = torch.load(HSI_PATH, weights_only=False)
gt = torch.load(GT_PATH, weights_only=False)
dataset_hsi = HsiDataset(data_hsi, gt, small_batches=IF_SMALL_BATCHES)
data_loader_hsi = DataLoader(dataset_hsi, batch_size=32, shuffle=True)
encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
encoder.conv1 = torch.nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder = encoder.to(CUDA0)
for data, label in data_loader_hsi:
    data = torch.permute(data, (0, 3, 1, 2))  # prepose the channel dim
    features = encoder(data)
    print(features.shape)
