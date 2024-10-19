import torch
from utils import util
import torch.utils.data as data

"""
HSI Size : (50, 4172, 1202)
RGB Size : (47680, 12020, 3)
Lidar Raster : (8344, 2404)
"""

HSI_PATH = '../data/FullHSIDataset/20170218_UH_CASI_S4_NAD83.txt'
rgb_root = '../data/Final RGB HR Imagery/'
rgb_list = ['UH_NAD83_272056_3290290.tif',
            'UH_NAD83_272652_3290290.tif',
            'UH_NAD83_273248_3290290.tif',
            'UH_NAD83_273844_3290290.tif']
LIDAR_RASTER_PATH = '../data/Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.txt'
BASE_PATH = '../data/Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.txt'
GROUND_TRUTH_PATH = '../data/Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.txt'


class DatasetFromTensor(data.Dataset):
    def __init__(self, tensor, gt, dataset_name):
        super(DatasetFromTensor, self).__init__()
        self.data = tensor
        self.gt = gt
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.data.shape


hsi = util.load_hsi_narray(HSI_PATH)

rgb = util.load_rgb_array(rgb_root, rgb_list)

lidar_raster = util.load_lidar_raster(LIDAR_RASTER_PATH)
base = util.base_loader(BASE_PATH)
ndsm = lidar_raster - base
gt = util.ground_truth_loader(GROUND_TRUTH_PATH)

for item in [hsi, rgb, ndsm, gt]:
    print(item.shape)
