from config import *
from discriminator import Discriminator
from generator import Generator
from utils import get_data, weights_init, save_models, n_params

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm.auto import trange

os.makedirs(sample_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# define the generator and discriminator
generator = Generator(latent_dim, out_dim, n_channels).to(device)
discriminator = Discriminator(n_channels).to(device)

print('----- Started GAN training -----')
print(f'---> Using {device}')
print(f'---> Generator contains {generator.n_params} weights')
print(f'---> Discriminator contains {discriminator.n_params} weights')

# initialize the weights of the discriminator
discriminator.apply(weights_init)

# define the loss function
criterion = nn.BCEWithLogitsLoss()

# define the optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# load the data
dataloader = get_data(dataroot, transform, batch_size)

# define the training loop
progress_bar = trange(1, n_iters+1)
for i in progress_bar:
    real_images, _ = next(dataloader)
    batch_size = real_images.shape[0]
    real_labels = torch.full((batch_size,), 1.0, device=device)
    fake_labels = torch.full((batch_size,), 0.0, device=device)

    # train the discriminator
    discriminator.zero_grad()
    real_images = real_images.to(device)
    real_outputs = discriminator(real_images).view(-1)
    real_loss = criterion(real_outputs, real_labels)
    real_loss.backward()

    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    fake_outputs = discriminator(fake_images.detach()).view(-1)
    fake_loss = criterion(fake_outputs, fake_labels)
    fake_loss.backward()

    d_loss = real_loss + fake_loss
    optimizer_d.step()

    # train the generator
    generator.zero_grad()
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    outputs = discriminator(fake_images).view(-1)
    g_loss = criterion(outputs, real_labels)
    g_loss.backward()
    optimizer_g.step()

    progress_bar.set_description(f'[D loss: {d_loss.item():.3f}] [G loss: {g_loss.item():.3f}]')

    if (i % 100) == 0:
        z = torch.randn(64, latent_dim, device=device)
        fake_images = generator(z).detach().cpu()

        # save the generated images
        save_path = os.path.join(sample_path, f'generated_{i:06}.png')
        vutils.save_image(fake_images, save_path, nrow=8, normalize=True)

        # save the models
        save_models(generator, discriminator, model_path, i=i)