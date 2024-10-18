import torch
import numpy as np
import cv2
HSI_SHAPE = (50, 4172, 1202)   # (band, width, height)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()


def load_hsi_narray(path):
    with open(path, 'r') as file:
        # [:223] is the description of the data
        data_string = file.read()[223:]
        data_array = np.array(list(map(int, data_string.split())), dtype=int)  # transfer str to np array
        data_array = data_array.reshape(HSI_SHAPE)
        data_array = data_array[:, 596:2980, 601:1202]
    data_array = torch.from_numpy(data_array).to(DEVICE)
    return data_array


def load_rgb_array(root, list):
    vhr = []
    for i in range(len(list)):
        img = np.array(cv2.imread(root + list[i], cv2.IMREAD_UNCHANGED))
        img = np.transpose(img, (2, 1, 0))
        vhr.append((img))
    vhr = np.concatenate(vhr, axis=1)
    vhr = torch.from_numpy(vhr).to(DEVICE)
    return vhr


def base_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(float, data_string.split())))
        base = np.reshape(base, (8344, 2404))
        base = torch.from_numpy(base).to(DEVICE)
    return base


def load_lidar_raster(path):
    count = 0
    with open(path, 'r') as file:
        data_strings = file.readlines()
        raster = np.array(list(map(float, data_strings[0].split())))[:, np.newaxis]
        for line in data_strings[1:-1]:
            data_array = np.array(list(map(float, line.split())))
            data_array = data_array[:, np.newaxis]
            # print(raster.shape)
            raster = np.concatenate((raster, data_array), axis=1)
        # data_array = np.array(list(map(float, data_string.split())))
        raster = torch.from_numpy(raster).to(DEVICE)
    return raster

