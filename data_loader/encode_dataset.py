import os
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm
from utils.params import DATA_ROOT, BATCH_SIZE, CUDA0
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights


class Dataset_from_data(data.Dataset):
    def __init__(self, root, stride=(2, 2), patch_size=(32, 32)):
        super(Dataset_from_data, self).__init__()
        self.root = root
        self.hsi, self.ndsm, self.rgb, self.gt = self._getdata()
        self.stride = stride
        self.patch_size = patch_size
        self.patches = self._generate_patches()

    def _getdata(self):
        hsi = torch.load(self.root + '/hsi.pth', weights_only=False)
        ndsm = torch.load(self.root + '/ndsm.pth', weights_only=False).float()
        rgb = torch.load(self.root + '/rgb.pth', weights_only=False).float()
        gt = torch.load(self.root + '/gt.pth', weights_only=False)
        return hsi, ndsm, rgb, gt

    def _generate_patches(self):
        count = 0
        patches = []
        stride_x, stride_y = self.stride
        patch_width, patch_height = self.patch_size
        width, height = self.hsi.shape[:2]
        for y in range(0, height - patch_height + 1, stride_y):
            patch = {}
            for x in range(0, width - patch_width + 1, stride_x):
                patch['hsi'] = self.hsi[x:x + patch_width, y:y + patch_height, :]
                patch['ndsm'] = self.ndsm[x:x + patch_width, y:y + patch_height, :]
                patch['rgb'] = self.rgb[x:x + patch_width, y:y + patch_height, :]
                patch['gt'] = self.gt[x:x + patch_width, y:y + patch_height]
                patches.append(patch)
                count += 1
        return patches

    def __getitem__(self, item):
        return self.patches[item]

    def __len__(self):
        return len(self.patches)


def get_modalities(patch):
    hsi, ndsm, rgb, gt = patch['hsi'].to(CUDA0), patch['ndsm'].to(CUDA0), patch['rgb'].to(CUDA0), patch['gt'].to(CUDA0)
    permuted_tensors = [tensor.permute(0, 3, 1, 2) for tensor in [hsi, ndsm, rgb]]
    hsi, ndsm, rgb = permuted_tensors
    return hsi, ndsm, rgb, gt


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        # Ensure that the input tensor size matches the product of new_sthape
        batch_size = x.size(0)
        num_elements = 1
        for dim in self.new_shape:
            num_elements *= dim

        if x.shape[1] != num_elements:
            raise ValueError("Total number of elements must be the same after reshape")
        return x.view(batch_size, *self.new_shape)


class MyEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MyEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = self.gen_encoder()
        self.reshape = Reshape((1, 32, 32))

    def gen_encoder(self):
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        encoder.conv1 = torch.nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        encoder.fc = torch.nn.Linear(2048, 1024, bias=True)
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.reshape(x)
        return x


def GenerateEncoders(option=0):
    if option == 0:
        print('No encoder generated!')
    elif option == 1:
        encoder1 = MyEncoder(48)
        encoder1 = encoder1.to(CUDA0)
        return encoder1
    elif option == 2:
        encoder2 = MyEncoder(1)
        encoder2 = encoder2.to(CUDA0)
        return encoder2
    elif option == 3:
        encoder3 = MyEncoder(3)
        encoder3 = encoder3.to(CUDA0)
        return encoder3


def encode_modalities(dataloader, encoder, option=0, save_dir='./encoded_features'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loop = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for step, patch in loop:
        hsi, ndsm, rgb, gt = get_modalities(patch)
        if option == 1:
            data = hsi
            del ndsm, rgb, gt
            torch.cuda.empty_cache()
        elif option == 2:
            data = ndsm
            del hsi, rgb, gt
            torch.cuda.empty_cache()
        elif option == 3:
            data = rgb
            del hsi, ndsm, gt
            torch.cuda.empty_cache()
        elif option == 4:
            data = gt
            del hsi, ndsm, rgb
            # filename = os.path.join(save_dir, f'encoded_data_{step}.pth')
            # torch.save(data, filename)
        else:
            raise ValueError("Option must be a figure of 1, 2, 3 or 4.")

        encoded_data = data

        filename = os.path.join(save_dir, f'encoded_data_{step}.pth')
        torch.save(encoded_data, filename)

        del encoded_data
        torch.cuda.empty_cache()
    loop.close()
    features = []
    for i in range(len(dataloader)):
        filename = os.path.join(save_dir, f'encoded_data_{i}.pth')
        feature = torch.load(filename)
        features.append(feature)
        os.remove(filename)
    return features


encoder_hsi = GenerateEncoders(1)
# encoder_ndsm = GenerateEncoders(2)
# encoder_rgb = GenerateEncoders(3)
raw_dataset = Dataset_from_data('.'+DATA_ROOT)
raw_data_loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False)
features_gt = encode_modalities(raw_data_loader, encoder_hsi, 4)
torch.save(features_gt, '../data/tensor/features/gt.pth')
print(f'Done!')

