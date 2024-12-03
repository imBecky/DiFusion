import torch.utils.data as data
import torch

SMALL_PATCH_NUM = 992


class DatasetFromTensor(data.Dataset):
    def __init__(self, hsi, ndsm, rgb, labels,
                 encoder_hsi, encoder_ndsm, encoder_rgb,
                 stride=(2, 2), patch_size=(32, 32), transform=None, small_batches=False):
        super(DatasetFromTensor, self).__init__()
        self.hsi = hsi.float()
        self.ndsm = ndsm.float()
        self.rgb = rgb.float()
        self.labels = labels
        self.stride = stride
        self.patch_size = patch_size
        self.transformation = transform
        self.small_batches = small_batches
        self.patches = self._generate_patches()
        self.encoder_hsi = encoder_hsi
        self.encoder_ndsm = encoder_ndsm
        self.encoder_rgb = encoder_rgb
        self.hsi, self.ndsm, self.rgb = self.generate_features()

    def generate_features(self):
        feature_hsi = self.encoder_hsi(self.hsi)
        feature_ndsm = self.encoder_ndsm(self.ndsm)
        feature_rgb = self.encoder_rgb(self.rgb)
        torch.save(feature_hsi, '../data/tensor/hsi.pth')
        torch.save(feature_ndsm, '../data/tensor/ndsm.pth')
        torch.save(feature_rgb, '../data/tensor/rgb.pth')
        print('encoding done!')
        return feature_hsi, feature_ndsm, feature_rgb

    def _generate_patches(self):
        count = 0
        patches = []
        stride_x, stride_y = self.stride
        patch_width, patch_height = self.patch_size
        width, height = self.hsi.shape[:2]
        for y in range(0, height-patch_height+1, stride_y):
            patch = {}
            if self.small_batches is True and count >= SMALL_PATCH_NUM:
                break
            for x in range(0, width-patch_width+1, stride_x):
                if self.small_batches is True and count >= SMALL_PATCH_NUM:
                    break
                patch['hsi'] = self.hsi[x:x + patch_width, y:y + patch_height, :]
                patch['ndsm'] = self.ndsm[x:x + patch_width, y:y + patch_height, :]
                patch['rgb'] = self.rgb[x:x + patch_width, y:y + patch_height, :]
                patch['label'] = torch.tensor(self.labels[x:x + patch_width, y:y + patch_height], dtype=torch.float)
                patches.append(patch)
                count += 1
        return patches

    def __getitem__(self, item):
        return self.patches[item]

    def __len__(self):
        return len(self.patches)

