import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class EncodingDataset(data.Dataset):
    def __init__(self, inputs, stride=(20, 20), patch_size=(64, 64), transform=None):
        """
        TODO: not sure about the stride
        """
        super(EncodingDataset, self).__init__()
        self.data = inputs
        self.transform = transform
        self.stride = stride
        self.patch_size = patch_size
        self.patches = self._generate_patches()

    def _generate_patches(self):
        patches = []
        height, width = self.data.shape[:2]
        patch_height, patch_width = self.patch_size
        stride_y, stride_x = self.stride
        for y in range(0, height-patch_height+1, stride_y):
            for x in range(0, width-patch_width+1, stride_x):
                patch = self.data[y:y+patch_height, x:x+patch_width, :]
                patches.append(patch)
        return patches

    def __getitem__(self, item):
        return self.patches[item]

    def __len__(self):
        return len(self.patches)


class FeatureDataset(data.Dataset):
    def __init__(self, input_path, gt_path):
        super(FeatureDataset, self).__init__()
        self.input = torch.load(input_path, weights_only=True)
        self.label = torch.load(gt_path, weights_only=True)

    def __getitem__(self, item):
        return self.input[item], self.label[item]

    def __len__(self):
        return len(self.label)