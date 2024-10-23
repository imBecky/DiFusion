"""
TODO: resnet50 can be replaced by other pre-trained model
"""
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from data_loader.dataset import EncodingDataset
HSI_PATH = './data/tensor/hsi.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.load(HSI_PATH, weights_only=True)
print(t)
raw_dataset_hsi = EncodingDataset(HSI_PATH)
print(len(raw_dataset_hsi))  # 这应该打印出数据集的大小，如果不是0，说明数据集加载成功
raw_data_loader_hsi = DataLoader(raw_dataset_hsi, batch_size=32, shuffle=True)


model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
model.conv1 = torch.nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)
data_loader = raw_data_loader_hsi
for batch in data_loader:
    print(batch.shape)
    batch = torch.permute(batch, (0, 3, 1, 2))
    print(batch.shape)
    output = model(batch)
    print(output.shape)
