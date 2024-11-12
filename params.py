import torch
DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_EPOCH = 30
BATCH_SIZE = 8
LEARNING_RATE1 = 0.005
LEARNING_RATE2 = 0.005
LEARNING_RATE3 = 0.005
T = 1000
image_size = 32
feature_channels = 1
IF_SMALL_BATCHES = True
dim_mults = (1, 2, 4,)