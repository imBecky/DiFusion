import torch
DATA_ROOT = './data/tensor'
GT_PATH = './data/tensor/gt.pth'
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 13
CLS_EPOCH = 3
BATCH_SIZE = 8
LEARNING_RATE1 = 0.0005  # lr for noise predictor
LEARNING_RATE2 = 0.0005  # lr for discriminator
LEARNING_RATE3 = 0.0005  # lr for classifier
T = 1000
image_size = 32
feature_channels = 1
IF_SMALL_DATASET = True
dim_mults = (1, 2, 4,)
