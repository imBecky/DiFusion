import os
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import torch
from utils.params import IF_SMALL_DATASET


class Dataset_from_feature(data.Dataset):
    def __init__(self, root, stride=(2, 2)):
        super(Dataset_from_feature, self).__init__()
        self.root = root
        self.stride = stride
        self.hsi, self.ndsm, self.rgb, self.gt = self._get_data()

    def _get_data(self):
        if IF_SMALL_DATASET:
            root = self.root+'/small'
            gt = torch.load(root + '/gt.pth', weights_only=False)
            gt = torch.from_numpy(gt)
            gt = torch.unsqueeze(gt, 1)
            feature_hsi = torch.load(root + '/hsi.pth', weights_only=False)
            feature_ndsm = torch.load(root + '/ndsm.pth', weights_only=False)
            feature_rgb = torch.load(root + '/rgb.pth', weights_only=False)
            for item in [feature_hsi, feature_ndsm, feature_rgb, gt]:
                print(item.shape)
            return feature_hsi, feature_ndsm, feature_rgb, gt
        else:
            root = self.root + '/feature'
            gt = torch.load(root + '/gt.pth', weights_only=False)
            feature_hsi = torch.load(root + '/hsi.pth', weights_only=False)
            feature_ndsm = torch.load(root + '/ndsm.pth', weights_only=False)
            feature_rgb = torch.load(root + '/rgb.pth', weights_only=False)
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