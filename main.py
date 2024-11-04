from model.Block1 import *
# from model.NosePredictor import GaussianDiffusion
import torch

DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
T = 1000
IF_SMALL_BATCHES = True

dataset_hsi, dataset_ndsm, dataset_rgb = GenerateDatasets(DATA_ROOT)
data_loader_hsi_train, data_loader_hsi_test = SpliteDataset(dataset_hsi, BATCH_SIZE, 0.8)
data_loader_ndsm_train, data_loader_ndsm_test = SpliteDataset(dataset_ndsm, BATCH_SIZE, 0.8)
data_loader_rgb_train, data_loader_rgb_test = SpliteDataset(dataset_rgb, BATCH_SIZE, 0.8)

encoder_hsi = GenerateEncoders(1)
encoder_ndsm = GenerateEncoders(2)
encoder_rgb = GenerateEncoders(3)

classifier = Classifier().to(CUDA0)
criterion = nn.CrossEntropyLoss().to(CUDA0)
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

Train(data_loader_hsi_train, encoder_hsi, classifier, T, criterion, optimizer, CLS_EPOCH)
# Train(data_loader_ndsm_train, encoder_ndsm, classifier, criterion, optimizer, CLS_EPOCH)
# Train(data_loader_rgb_train, encoder_rgb, classifier, criterion, optimizer, CLS_EPOCH)
# Test(data_loader_rgb_test, encoder_rgb, classifier)


