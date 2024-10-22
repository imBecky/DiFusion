import torch
from utils import util
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


hsi = util.load_hsi_narray(HSI_PATH)
torch.save(hsi, '../data/tensor/hsi.pth')

rgb = util.load_rgb_array(rgb_root, rgb_list)
torch.save(rgb, '../data/tensor/rgb.pth')

lidar_raster = util.load_lidar_raster(LIDAR_RASTER_PATH)
base = util.base_loader(BASE_PATH)
ndsm = lidar_raster - base
torch.save(ndsm, '../data/tensor/ndsm.pth')
gt = util.ground_truth_loader(GROUND_TRUTH_PATH)
torch.save(gt, '../data/tensor/gt.pth')

for item in [hsi, rgb, ndsm, gt]:
    print(item.shape)
