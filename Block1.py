import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from data_loader.dataset import DatasetFromTensor

DATA_ROOT = './data/tensor'
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
    """TODO: deepen the model"""
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


def GenerateDatasets(root):
    data_hsi = torch.load(root + '/hsi.pth', weights_only=False)
    data_ndsm = torch.load(root + '/ndsm.pth', weights_only=False)
    data_rgb = torch.load(root + '/rgb.pth', weights_only=False)
    gt = torch.load(root + '/gt.pth', weights_only=False)

    dataset_hsi = DatasetFromTensor(data_hsi, gt, small_batches=IF_SMALL_BATCHES)
    dataset_ndsm = DatasetFromTensor(data_ndsm, gt, small_batches=IF_SMALL_BATCHES)
    dataset_rgb = DatasetFromTensor(data_rgb, gt, small_batches=IF_SMALL_BATCHES)
    return dataset_hsi, dataset_ndsm, dataset_rgb


def GenerateDataLoaders(hsi, ndsm, rgb):
    data_loader_hsi = DataLoader(hsi,  batch_size=32, shuffle=True)
    data_loader_ndsm = DataLoader(ndsm, batch_size=32, shuffle=True)
    data_loader_rgb = DataLoader(rgb, batch_size=32, shuffle=True)
    return data_loader_hsi, data_loader_ndsm, data_loader_rgb


def GenerateEncoders(option=0):
    if option == 0:
        print('No encoder generated!')
    elif option == 1:
        encoder1 = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        encoder1.conv1 = torch.nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder1 = encoder1.to(CUDA0)
        return encoder1
    elif option == 2:
        encoder2 = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        encoder2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder2 = encoder2.to(CUDA0)
        return encoder2


def Train(dataloader, encoder, classifier, criterion, optimizer, epoch_num):
    for epoch in range(epoch_num):
        running_loss = 0.0
        for data, label in dataloader:
            data, label = data.to(CUDA0), label.to(CUDA0)
            data = torch.permute(data, (0, 3, 1, 2))  # prepose the channel dim
            features = encoder(data)
            output = classifier(features)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader_hsi)}')


dataset_hsi, dataset_ndsm, dataset_rgb = GenerateDatasets(DATA_ROOT)
data_loader_hsi, data_loader_ndsm, data_loader_rgb = GenerateDataLoaders(dataset_hsi, dataset_ndsm, dataset_rgb)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
classifier = Classifier().to(CUDA0)
criterion = nn.CrossEntropyLoss().to(CUDA0)
optimizer = optim.Adam(classifier.parameters(), lr=0.05)

# Train(data_loader_hsi, encoder_hsi, classifier, criterion, optimizer, CLS_EPOCH)
Train(data_loader_ndsm, encoder_ndsm, classifier, criterion, optimizer, CLS_EPOCH)
