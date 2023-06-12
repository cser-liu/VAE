"""
encoder of vae
create by liudan

"""

import torch
import torch.nn as nn

class encoder(nn.Module):
    """
    input : batchsize x 3 x 64 x 64 
    output: batchsize x z_dim
    """
    def __init__(self, in_channels=3, z_dim=20, batch_size=32):
        super(encoder, self).__init__()
        self.in_channels  =in_channels
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, padding=1, stride=2)  #bs x 64 x 32 x 32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2) #bs x 128 x 16 x 16

        self.relu = nn.ReLU()

        self.fc11 = nn.Linear(in_features=128*16*16, out_features=1024)
        self.fc12 = nn.Linear(1024, z_dim)

        self.fc21 = nn.Linear(in_features=128*16*16, out_features=1024)
        self.fc22 = nn.Linear(1024, z_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(self.batch_size, -1) #bs x 128*16*16

        z_mu = self.relu(self.fc11(x))
        z_mu = self.fc12(z_mu)

        z_sigma = self.relu(self.fc21(x))
        z_sigma = self.fc22(z_sigma)

        return z_mu, z_sigma
        

        


