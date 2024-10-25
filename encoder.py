"""
TODO: resnet50 can be replaced by other pre-trained model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from data_loader.dataset import EncodingDataset
from data_loader.dataset import FeatureDataset

HSI_PATH = './data/tensor/hsi.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 20)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def encode_hsi(inputs, model):
    raw_dataset_hsi = EncodingDataset(inputs)
    raw_data_loader_hsi = DataLoader(raw_dataset_hsi, batch_size=32, shuffle=True)
    features = None
    count = 0
    for batch in raw_data_loader_hsi:
        count += 1
        batch = torch.permute(batch, (0, 3, 1, 2))  # prepose the channel dim
        feature = model(batch)
        print(f'{count}: HSI encode DONE, the shape of output is {feature.shape}')
        if features is not None:
            features = torch.cat((features, feature), dim=0)
        else:
            features = feature
    print(features.shape)
    return features


model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
model.conv1 = torch.nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(CUDA0)
classifier = Classifier().to(CUDA0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
inputs_hsi = torch.load(HSI_PATH, weights_only=True)
hsi_feature = encode_hsi(inputs_hsi, model)
# for epoch in range(CLS_EPOCH):
#     pass
