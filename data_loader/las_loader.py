import laspy
import os
import numpy as np
from operator import attrgetter
count = 0

"""the shape of data is 4172* 1202 = 501,4744
                                  203,302,725
the shape of each tile is 596*601
the shape of the whole pic is 2 tiles * 7
las data is classified according to points, not pixel
gt can be obtained through las.classification.array"""


LAS_ROOT = '../data/Lidar Point Cloud Tiles'
CHANNELS = ['C1', 'C2', 'C3', 'C123']
TILE_SHAPE = (4768, 1202)
las_list = ['273844_3290290.las',
            '273248_3290290.las',
            '272652_3290290.las',
            '272056_3290290.las']


def load_las_array(root, channel, data_list):
    las_data = None
    # for i in range(len(las_list)):
    for i in range(1):
        las_file = laspy.read(os.path.join(root, channel, data_list[i]))
        las_xyz = np.array(las_file.xyz, dtype=np.float32)
        las_gt = np.array(las_file.classification.array, dtype=int)
        las_gt = las_gt[:, np.newaxis]
        las_tile = np.concatenate((las_xyz, las_gt), axis=1)
        if i == 0:
            las_data = las_tile
        las_data = np.concatenate((las_data, las_tile), axis=0)
    base = np.array((min(las_data[:, 0]), min(las_data[:, 1]), min(las_data[:, 2]), 0))
    las_data = las_data - base
    las_data[:, :2] = las_data[:, :2] * 2
    # sort rows in ascending order of x, columns in ascending order of y
    keys = las_data[:, 0], las_data[:, 1]
    sort_index = np.lexsort(keys, axis=0)
    sorted_data = np.array(las_data[sort_index], dtype=np.float32)
    sorted_data[:, :2] = np.floor(sorted_data[:, :2])
    return sorted_data


def pixel_cluster(data):
    print(data[:10])
    cluster = np.array([[data[0]]])
    # print(cluster)
    for i in range(1, data.shape[0]):
        item = data[i]
        i_xy = item[:2]
        present_xy = cluster[-1, 0, :2]
        # print(i_xy, '\n', present_xy)
        if np.array_equal(i_xy, present_xy):
            print('1111')
        else:
            np.append(cluster, item)
            print(cluster)
        if i == 3:
            break


def data_report(data):
    max_point_len = 0
    max_xy = np.array([1, 0])
    tempt_len = 0
    tempt_xy = np.array([1, 0])
    for i in range(data.shape[0]):
        x = int(data[i, 0])
        y = int(data[i, 1])
        xy = np.array([x, y])
        if np.array_equal(xy, tempt_xy):
            tempt_len += 1
            if tempt_len > max_point_len:
                max_point_len = tempt_len
                max_xy = xy
        else:
            tempt_xy = xy
            tempt_len = 1
    print(max_point_len, '\n', max_xy)


data = load_las_array(LAS_ROOT, CHANNELS[0], las_list)
# pixel_cluster(data)
data_report(data)


