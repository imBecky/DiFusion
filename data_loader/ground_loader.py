import numpy as np

GROUND_TRUTH_PATH = '../data/Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.txt'


def edit_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    with open(path, 'w') as file:
        for line in lines:
            if ';' not in line:
                file.write(line)


def ground_truth_loader(path):
    with open(path, 'r') as file:
        data_string = file.read()
        base = np.array(list(map(float, data_string.split())))
        base = np.reshape(base, (8344, 2404))
        return base


edit_file(GROUND_TRUTH_PATH)
base = ground_truth_loader(GROUND_TRUTH_PATH)
print(base)
