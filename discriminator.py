import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()

        self.n_channels = n_channels

        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.AdaptiveAvgPool1d(512),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def load(filename):
        '''
        Loads model weights and configuration
        '''
        state_dict = torch.load(filename)
        model = Discriminator(**state_dict['args'])
        model.load_state_dict(state_dict['state_dict'])
        return model

    def save(self, filename):
        '''
        Saves model weights and configuration
        '''
        state_dict = {
            'args': {
                'n_channels': self.n_channels
            },
            'state_dict': self.state_dict()
        }
        torch.save(state_dict, filename)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
