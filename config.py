'''
Constants and hyperparameters :)
'''

import torch
from torchvision import transforms

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
dataroot = './data' # 'path/to/dataset'
sample_path = './samples'
model_path = './models'
n_channels = 3
n_iters = 1000
batch_size = 32
latent_dim = 100
out_dim = 16 # 128
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
epochs = 50

transform = transforms.Compose([
    transforms.Resize((out_dim, out_dim)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    )
])