"""
TODO: resnet50 can be replaced by other pre-trained model
"""
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from data_loader.dataset import EncodingDataset
HSI_PATH = './data/tensor/hsi.pth'

raw_data_loader_hsi = EncodingDataset(HSI_PATH)


model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
data_loader = raw_data_loader_hsi
for batch in data_loader:
    output = model(batch)
    print(output.shape)
