import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from data_loader.old_dataset import DatasetFromTensor
import tqdm
# from .utils.util import calculate_fid
from utils.params import *


def get_modalities(patch):
    hsi, ndsm, rgb, label = patch['hsi'].to(CUDA0), patch['ndsm'].to(CUDA0), \
                            patch['rgb'].to(CUDA0), patch['label'].to(CUDA0)
    permuted_tensors = [tensor.permute(0, 3, 1, 2) for tensor in [hsi, ndsm, rgb]]
    hsi, ndsm, rgb = permuted_tensors
    return hsi, ndsm, rgb, label


def noise_predictor_trainer(GaussianDiffuser, t,
                            feature_hsi, feature_ndsm, feature_rgb):
    losses = []
    noised_features = []
    for i, name in enumerate(['hsi', 'ndsm', 'rgb']):
        feature = locals()['feature_' + name]
        optimizer = getattr(GaussianDiffuser, 'noise_predictor_optimizer_' + name)
        noise_predictor = getattr(GaussianDiffuser, 'noise_predictor_' + name)
        optimizer.zero_grad()
        noise = torch.randn_like(feature).to(CUDA0)
        X_t = GaussianDiffuser.diffuse(feature, t, noise)
        noised_features.append(X_t.detach())
        predicted_noise = noise_predictor(X_t, t)
        loss_i = GaussianDiffuser.noise_predictor_criterion(predicted_noise, noise)
        losses.append(loss_i)
        loss_i.backward()
        optimizer.step()
    return noised_features, losses


def generate_feature(GaussianDiffuser, name, noised_x_t, t):
    predictor = getattr(GaussianDiffuser, 'noise_predictor_' + name)
    noise_hat = predictor(noised_x_t, t)
    feature_hat = GaussianDiffuser.generate(noised_x_t.shape, noise_hat, t)
    return feature_hat


def block2(GaussianDiffuser, t,
           feature_hsi, feature_ndsm, feature_rgb, label,
           noised_hsi, noised_ndsm, noised_rgb):
    # Train the discriminator
    d_loss = torch.tensor(0.).to(CUDA0)
    GaussianDiffuser.discriminator_optimizer.zero_grad()
    for i, name in enumerate(['hsi', 'ndsm', 'rgb']):
        noised_x_t = locals()['noised_' + name]
        feature_hat = generate_feature(GaussianDiffuser, name, noised_x_t, t)
        d_labels = torch.full((BATCH_SIZE, 1), i, dtype=torch.float32).to(CUDA0)
        output_fake = GaussianDiffuser.discriminator(feature_hat.detach().to(CUDA0))
        d_loss_i = GaussianDiffuser.discriminator_criterion(output_fake, d_labels)
        d_loss += d_loss_i
    d_loss /= 3
    d_loss.backward()
    GaussianDiffuser.discriminator_optimizer.step()
    g_loss = []
    for i, name in enumerate(['hsi', 'ndsm', 'rgb']):
        optimizer = getattr(GaussianDiffuser, 'noise_predictor_optimizer_' + name)
        optimizer.zero_grad()
        noised_x_t = locals()['noised_' + name]
        g_labels = torch.full((BATCH_SIZE,), i, dtype=torch.float32).to(CUDA0)
        feature_hat = generate_feature(GaussianDiffuser, name, noised_x_t, t)
        outputs_fake = GaussianDiffuser.discriminator(feature_hat)
        g_loss.append(GaussianDiffuser.generate_criterion(outputs_fake, g_labels))
        g_loss[i].backward(retain_graph=(i != 2))
        optimizer.step()
    return d_loss, g_loss


def Train(dataloader_train, GaussianDiffuser, epoch_num):
    loop = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))
    feature_hsi, feature_ndsm, feature_rgb, label = [], [], [], []
    for step, patch in loop:
        hsi, ndsm, rgb, label = get_modalities(patch)
        feature_hsi, feature_ndsm, feature_rgb = encode_modalities(hsi, ndsm, rgb, GaussianDiffuser)
    del hsi, ndsm, rgb
    del GaussianDiffuser.encoder_ndsm, GaussianDiffuser.encoder_hsi, GaussianDiffuser.encoder_rgb
    for epoch in range(epoch_num):
        running_classification_loss = 0.0
        fid_score = -0.0
        for step, patch in loop:
            # del step, patch
            torch.cuda.empty_cache()
            t = torch.randint(0, T, (BATCH_SIZE,), device=CUDA0).long()
            # train the noise predictor
            noised_features, noise_losses = noise_predictor_trainer(GaussianDiffuser, t,
                                                                    feature_hsi, feature_ndsm, feature_rgb)
            noised_hsi, noised_ndsm, noised_rgb = noised_features
            block2(GaussianDiffuser, t, feature_hsi, feature_ndsm, feature_rgb, label,
                   noised_hsi, noised_ndsm, noised_rgb)
            # running_classification_loss += classification_loss_hsi
            # fid_score = calculate_fid(noise_hsi.cpu().detach().numpy(), noise_hsi_hat.cpu().detach().numpy())
        # print(f'Epoch {epoch + 1}, Loss: {running_classification_loss / len(dataloader_train)}, FID: {fid_score}')


def Test(dataloader_test, encoder, classifier):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(CUDA0), labels.to(CUDA0)
            inputs = torch.permute(inputs, (0, 3, 1, 2))  # prepose the channel dim
            feature = encoder(inputs)
            predicted = classifier(feature).round()
            equal_elements = torch.eq(predicted, labels)
            count = torch.sum(equal_elements)
            batch_num = 1
            for i in labels.shape:
                batch_num *= i
            total += batch_num
            correct += count
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
