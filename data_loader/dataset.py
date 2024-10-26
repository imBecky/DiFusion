import torch.utils.data as data
import torch

SMALL_PATCH_NUM = 1000


class HsiDataset(data.Dataset):
    def __init__(self, inputs, labels, stride=(20, 20), patch_size=(64, 64), transform=None, small_batches=False):
        super(HsiDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels
        self.stride = stride
        self.patch_size = patch_size
        self.transformation = transform
        self.small_batches = small_batches
        self.patches = self._generate_patches()

    def _generate_patches(self):
        count = 0
        patches = []
        stride_x, stride_y = self.stride
        patch_width, patch_height = self.patch_size
        width, height = self.inputs.shape[:2]
        for x in range(0, width-patch_width+1, stride_x):
            if self.small_batches is True and count == SMALL_PATCH_NUM:
                break
            for y in range(0, height-patch_height+1, stride_y):
                patch = self.inputs[x:x + patch_width, y:y + patch_height, :], \
                        self.labels[x:x + patch_width, y:y + patch_height]
                patches.append(patch)
                count += 1
            return patches

    def __getitem__(self, item):
        return self.patches[item]

    def __len__(self):
        return len(self.patches)


