import tqdm
from utils.params import *


def get_modalities(patch):
    hsi, ndsm, rgb, label = patch[0].to(CUDA0), patch[1].to(CUDA0), \
                            patch[2].to(CUDA0), patch[3].to(CUDA0)
    # permuted_tensors = [tensor.permute(0, 3, 1, 2) for tensor in [hsi, ndsm, rgb]]
    # hsi, ndsm, rgb = permuted_tensors
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


def Generate(GaussianDiffuser, name, noised_x_t, t):
    predictor = getattr(GaussianDiffuser, 'noise_predictor_' + name)
    noise_hat = predictor(noised_x_t, t)
    feature_hat = GaussianDiffuser.generate(noised_x_t.shape, noise_hat, t)
    return feature_hat


def Generate_n_Discriminate(GaussianDiffuser, t,
                            feature_hsi, feature_ndsm, feature_rgb, label,
                            noised_hsi, noised_ndsm, noised_rgb):
    # Train the discriminator
    d_loss = torch.tensor(0.).to(CUDA0)
    GaussianDiffuser.discriminator_optimizer.zero_grad()
    GaussianDiffuser.classifier_optimizer.zero_grad()
    for i, name in enumerate(['hsi', 'ndsm', 'rgb']):
        noised_x_t = locals()['noised_' + name]
        feature_hat = Generate(GaussianDiffuser, name, noised_x_t, t)
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
        feature_hat = Generate(GaussianDiffuser, name, noised_x_t, t)
        outputs_fake = GaussianDiffuser.discriminator(feature_hat)
        g_loss.append(GaussianDiffuser.generate_criterion(outputs_fake, g_labels))
        g_loss[i].backward(retain_graph=(i != 2))
        optimizer.step()
    return d_loss, g_loss


def Train(dataloader_train, GaussianDiffuser, epoch_num):
    for i in range(epoch_num):
        losses = []
        loop = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for loop, patch in loop:
            hsi, ndsm, rgb, label = get_modalities(patch)
            t = torch.randint(0, T, (BATCH_SIZE,), device=CUDA0).long()
            noised_features, noise_losses = noise_predictor_trainer(GaussianDiffuser, t, hsi, ndsm, rgb)
            noised_hsi, noised_ndsm, noised_rgb = noised_features
            Generate_n_Discriminate(GaussianDiffuser, t, hsi, ndsm, rgb, label,
                                    noised_hsi, noised_ndsm, noised_rgb)
