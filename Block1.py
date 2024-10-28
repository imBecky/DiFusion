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
CLS_EPOCH = 20
IF_SMALL_BATCHES = True


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        # Ensure that the input tensor size matches the product of new_sthape
        batch_size = x.size(0)
        num_elements = 1
        for dim in self.new_shape:
            num_elements *= dim

        if x.shape[1] != num_elements:
            raise ValueError("Total number of elements must be the same after reshape")
        return x.view(batch_size, *self.new_shape)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 16*16),
            nn.ReLU(),
            nn.BatchNorm1d(16*16),
            nn.Dropout(0.5),
            nn.Linear(16*16, 32*32),
            nn.ReLU(),
            Reshape((32, 32))
        )

    def forward(self, x):
        x = self.model(x)
        return x


data_hsi = torch.load(HSI_PATH, weights_only=False)
gt = torch.load(GT_PATH, weights_only=False)
dataset_hsi = HsiDataset(data_hsi, gt, small_batches=IF_SMALL_BATCHES)
data_loader_hsi = DataLoader(dataset_hsi, batch_size=32, shuffle=True)
encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
encoder.conv1 = torch.nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder = encoder.to(CUDA0)
classifier = Classifier().to(CUDA0)
criterion = nn.CrossEntropyLoss().to(CUDA0)
optimizer = optim.Adam(classifier.parameters(), lr=0.05)
for epoch in range(CLS_EPOCH):
    running_loss = 0.0
    for data, label in data_loader_hsi:
        data, label = data.to(CUDA0), label.to(CUDA0)
        data = torch.permute(data, (0, 3, 1, 2))  # prepose the channel dim
        features = encoder(data)
        output = classifier(features)
        # print(f'output:{output[:15]}')
        # print(f'label:{label[:15]}')
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader_hsi)}')
