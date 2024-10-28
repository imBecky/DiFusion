import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

HSI_PATH = '../data/tensor/hsi.pth'


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()


class TemptDataset(Dataset):
    def __init__(self, tensor, patch_size=(298, 601)):
        super(TemptDataset, self).__init__()
        self.data = tensor
        self.stride = patch_size
        self.patch_size = patch_size
        self.patches = self._generate_patches()

    def _generate_patches(self):
        patches = []
        for i in range(8):
            x = i*298
            patch = self.data[x:x+298, :, :]
            patches.append(patch)
        return patches

    def __getitem__(self, item):
        return self.patches[item]

    def __len__(self):
        return len(self.patches)


def resample_tensor(tensor):
    tensor = torch.permute(tensor, (0, 3, 1, 2)).float()
    resampled = F.interpolate(
        tensor, scale_factor=(2, 2), mode='bilinear', align_corners=False
    )
    resampled = torch.permute(resampled, (0, 2, 3, 1)).int()
    return resampled


def Step1(step1=False):
    hsi_tensor = torch.load(HSI_PATH, weights_only=False)
    hsi_tensor = hsi_tensor[:48, :, :]  # remove the description dim
    hsi_tensor = hsi_tensor.permute((1, 2, 0))
    # Now the shape of hsi tensor is supposed to be (2384, 601, 48)
    # torch.save(hsi_tensor, '../data/tensor/hsi.pth')
    if step1:
        tempt_hsi_dataset = TemptDataset(hsi_tensor)
        print('Dataset Generated!')
        tempt_data_loader_hsi = DataLoader(tempt_hsi_dataset, batch_size=1, shuffle=False)
        for idx, batch in enumerate(tempt_data_loader_hsi):
            resample_batch = resample_tensor(batch)
            torch.save(resample_batch, f'../data/tempt/{idx}.pth')


def Step2(step2=False):
    if step2:
        resampled_hsi = torch.tensor([]).to(DEVICE)
        for i in range(8):
            block = torch.load(f'../data/tempt/{i}.pth')
            resampled_hsi = torch.cat((resampled_hsi, block), dim=1)
        print(resampled_hsi.shape)
        torch.save(resampled_hsi, '../data/tensor/hsi.pth')


Step1(True)
Step2(True)
t = torch.load('../data/tensor/hsi.pth')
t = t[0]
torch.save(t, '../data/tensor/hsi.pth')
