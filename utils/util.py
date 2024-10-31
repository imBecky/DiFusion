import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
HSI_SHAPE = (50, 4172, 1202)   # (band, width, height)
new_shape = (50, 8344, 2404)


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
            indices = np.where(data_array == 1000)
            if len(indices[0]) != 0:
                data_array[indices[0], 0] = -16
            raster = np.concatenate((raster, data_array), axis=1)
        # data_array = np.array(list(map(float, data_string.split())))
        raster = torch.from_numpy(raster).to(DEVICE)
        return raster


def ground_truth_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(int, data_string.split())))
        base = np.reshape(base, (4768, 1202))
        return base


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


class EMA:
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

