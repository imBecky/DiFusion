import rasterio
import rasterio.mask
import numpy as np

gt_path = '../data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.txt'

labels = ['Unclassified', 'Healthy grass', 'Stressed grass',
          'Artificial turf', 'Evergreen trees', 'Deciduous trees',
          'Bare earth', 'Water', 'Residential buildings',
          'Non-residential buildings', 'Roads', 'Sidewalks',
          'Crosswalks', 'Major thoroughfares', 'Highways',
          'Railways', 'Paved parking lots', 'Unpaved parking lots',
          'Cars', 'Trains', 'Stadium seats']


def load_gt(path):
    with open(path) as src:
        src_str = src.read()[:]
        gt_array = np.array(list(map(int, src_str.split())), dtype=int)
        print(gt_array.shape)
        gt_array = gt_array.reshape((4768, 1202))
        return gt_array


gt = load_gt(gt_path)
print(gt)
