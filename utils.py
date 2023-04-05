'''
Utilities :)
'''

from generator import Generator
import os
from switchblade.utils import infinite_dataloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_data(dataroot, transform, batch_size):
    dataset = ImageFolder(dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = infinite_dataloader(dataloader)
    return dataloader

def save_models(generator, discriminator, save_path, i=None):
    i = f'_{i:06}' if i is not None else ''
    g_path = os.path.join(save_path, f'generator{i}.pt')
    d_path = os.path.join(save_path, f'discriminator{i}.pt')

    generator.save(g_path)
    torch.save(discriminator.state_dict(), d_path)

def load_models(g_filename, d_filename):
    generator = Generator.load(g_filename)
    discriminator = resnet18(pretrained=False, num_classes=1)
    discriminator.load_state_dict(torch.load(d_filename))
    return generator, discriminator

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)