import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from data_loader.dataset import DatasetFromTensor
import tqdm
from .NoisePredictor import q_sample, generate
from utils.util import calculate_fid

DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
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


class MyEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MyEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = self.gen_encoder()
        self.reshape = Reshape((1, 32, 32))

    def gen_encoder(self):
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        encoder.conv1 = torch.nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        encoder.fc = torch.nn.Linear(2048, 1024, bias=True)
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.reshape(x)
        return x


def SpliteDataset(dataset, batch_size, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


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
        encoder1 = MyEncoder(48)
        encoder1 = encoder1.to(CUDA0)
        return encoder1
    elif option == 2:
        encoder2 = MyEncoder(1)
        encoder2 = encoder2.to(CUDA0)
        return encoder2
    elif option == 3:
        encoder3 = MyEncoder(3)
        encoder3 = encoder3.to(CUDA0)
        return encoder3


def Train(dataloader_train, encoder, noise_predictor, classifier, T, criterion, optimizer, epoch_num):
    for epoch in range(epoch_num):
        running_loss = 0.0
        loop = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for step, (datas, labels) in loop:
            datas, labels = datas.to(CUDA0), labels.to(CUDA0)
            datas = torch.permute(datas, (0, 3, 1, 2))    # pre_pose the channel dim (b, c, h, w)
            features = encoder(datas)
            batch_size = features.shape[0]
            t = torch.randint(0, T, (batch_size,), device=CUDA0).long()
            noise = torch.randn_like(features).to(CUDA0)
            noised = q_sample(features, t, noise)
            noise_hat = noise_predictor(noised, t)
            X_0_hat = generate(features.shape, noise_hat, t)
            print(X_0_hat.shape)
            loss = F.smooth_l1_loss(noise, noise_hat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            fid_score = calculate_fid(noise.cpu().detach().numpy(), noise_hat.cpu().detach().numpy())
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader_train)}, FID: {fid_score}')


def Test(dataloader_test, encoder, classifier):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(CUDA0), labels.to(CUDA0)
            inputs = torch.permute(inputs, (0, 3, 1, 2))  # prepose the channel dim
            feature = encoder(inputs)
            predicted = classifier(feature).round()
            equal_elements = torch.eq(predicted, labels)
            count = torch.sum(equal_elements)
            batch_num = 1
            for i in labels.shape:
                batch_num *= i
            total += batch_num
            correct += count
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

