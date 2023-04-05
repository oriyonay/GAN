import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim, n_channels=3):
        super().__init__()

        assert out_dim >= 16

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.n_channels = n_channels

        # Define the layers of the generator network
        self.fc1 = nn.Linear(latent_dim, 512 * (out_dim // 16) ** 2)
        self.bn1 = nn.BatchNorm1d(512 * (out_dim // 16) ** 2)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.upconv4 = nn.ConvTranspose2d(64, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        # Project and reshape the input noise
        x = self.bn1(self.fc1(z))
        x = F.relu(x)
        x = x.view(-1, 512, self.out_dim // 16, self.out_dim // 16)

        # Upsample and apply batch normalization and ReLU
        x = self.bn2(self.upconv1(x))
        x = F.relu(x)

        x = self.bn3(self.upconv2(x))
        x = F.relu(x)

        x = self.bn4(self.upconv3(x))
        x = F.relu(x)

        # Generate the output image
        x = torch.tanh(self.upconv4(x))

        return x
    
    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        imgs = self(z)
        pass

    @staticmethod
    def load(filename):
        '''
        Loads model weights and configuration
        '''
        state_dict = torch.load(filename)
        model = Generator(**state_dict['args'])
        model.load_state_dict(state_dict['state_dict'])
        return model

    def save(self, filename):
        '''
        Saves model weights and configuration
        '''
        state_dict = {
            'args': {
                'latent_dim': self.latent_dim,
                'out_dim': self.out_dim,
                'n_channels': self.n_channels
            },
            'state_dict': self.state_dict()
        }
        torch.save(state_dict, filename)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)