import tqdm
from utils.params import *


def get_modalities(patch):
    hsi, ndsm, rgb, label = patch[0].to(CUDA0), patch[1].to(CUDA0), \
                            patch[2].to(CUDA0), patch[3].to(CUDA0)
    permuted_tensors = [tensor.permute(0, 3, 1, 2) for tensor in [hsi, ndsm, rgb]]
    hsi, ndsm, rgb = permuted_tensors
    return hsi, ndsm, rgb, label


def Train(dataloader_train, GaussianDiffuser, epoch_num):
    loop = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))
    for step, patch in loop:
        hsi, ndsm, rgb, label = get_modalities(patch)
        for item in [hsi, ndsm, rgb, label]:
            print(item.shape)