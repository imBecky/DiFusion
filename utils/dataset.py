import os
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import torch
from utils.params import DATA_ROOT, BATCH_SIZE, IF_SMALL_DATASET, CUDA0


class Dataset_from_feature(data.Dataset):
    def __init__(self, root, stride=(2, 2)):
        super(Dataset_from_feature, self).__init__()
        self.root = root
        self.stride = stride
        self.hsi, self.ndsm, self.rgb, self.gt = self._get_data()

    def _get_data(self):
        if IF_SMALL_DATASET:
            gt = torch.load(os.path.dirname(self.root) + '/gt.pth', weights_only=False)[:20480]
            feature_hsi = torch.load(self.root + '/hsi.pth',
                                     weights_only=False, map_location=torch.device('cpu'))
            small_feature_hsi = feature_hsi[:20480]
            small_feature_hsi = small_feature_hsi.to(CUDA0)
            del feature_hsi

            feature_ndsm = torch.load(self.root + '/ndsm.pth',
                                      weights_only=False, map_location=torch.device('cpu'))
            small_feature_ndsm = feature_ndsm[:20480]
            small_feature_ndsm = small_feature_ndsm.to(CUDA0)
            del feature_ndsm

            feature_rgb = torch.load(self.root + '/rgb.pth',
                                     weights_only=False, map_location=torch.device('cpu'))
            small_feature_rgb = feature_rgb[:20480]
            small_feature_rgb = small_feature_rgb.to(CUDA0)
            del feature_rgb
            torch.cuda.empty_cache()
            for item in [small_feature_hsi, small_feature_ndsm, small_feature_rgb, gt]:
                print(item.shape)
            return small_feature_hsi, small_feature_ndsm, small_feature_rgb, gt
        else:
            gt = torch.load(os.path.dirname(self.root) + '/gt.pth', weights_only=False)
            feature_hsi = torch.load(self.root + '/hsi.pth', weights_only=False)
            feature_ndsm = torch.load(self.root + '/ndsm.pth', weights_only=False)
            feature_rgb = torch.load(self.root + '/rgb.pth', weights_only=False)
            return feature_hsi, feature_ndsm, feature_rgb, gt

    def __getitem__(self, item):
        return self.hsi[item], self.ndsm[item], self.rgb[item], self.gt[item]

    def __len__(self):
        return len(self.hsi)


def SpliteDataset(dataset, batch_size, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


dataset = Dataset_from_feature(DATA_ROOT+'/features')
data_loader_train, data_loader_test = SpliteDataset(dataset, BATCH_SIZE, 0.8)
