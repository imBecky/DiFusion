import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import torch
from params import DATA_ROOT, BATCH_SIZE


class Dataset_from_feature(data.Dataset):
    def __init__(self, root, stride=(2, 2), patch_size=(32, 32)):
        super(Dataset_from_feature, self).__init__()
        self.root = root
        self.stride = stride
        self.patch_size = patch_size
        self.hsi, self.ndsm, self.rgb = self._get_features()

    def _get_features(self):
        feature_hsi = torch.load(self.root+'/hsi.pth', weights_only=False)
        feature_ndsm = torch.load(self.root+'/ndsm.pth', weights_only=False)
        feature_rgb = torch.load(self.root+'/rgb.pth', weights_only=False)
        for item in [feature_hsi, feature_ndsm, feature_rgb]:
            print(item.shape)
        return feature_hsi, feature_ndsm, feature_rgb

    def __getitem__(self, item):
        return self.hsi[item], self.ndsm[item], self.rgb[item]

    def __len__(self):
        return len(self.hsi)


def SpliteDataset(dataset, batch_size, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


dataset = Dataset_from_feature('.'+DATA_ROOT+'/features')
data_loader_train, data_loader_test = SpliteDataset(dataset, BATCH_SIZE, 0.8)
