import numpy as np
import torch

LIDAR_RASTER_PATH = '../data/Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.txt'
BASE_PATH = '../data/Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.txt'


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()


def base_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(float, data_string.split())))
        base = np.reshape(base, (8344, 2404))
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
        return raster


lidar_raster = load_lidar_raster(LIDAR_RASTER_PATH)
lidar_raster = torch.from_numpy(lidar_raster).to(DEVICE)
base = base_loader(BASE_PATH)
base = torch.from_numpy(base).to(DEVICE)
ndsm = lidar_raster - base
print(ndsm)


