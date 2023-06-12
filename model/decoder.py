"""
decoder of vae
"""

import torch
import torch.nn as nn

class decoder(nn.Module):
    """
    input : batchsize x samples x z_dim
    output: batchsize x 3 x 64 x 64 
    """
    def __init__(self, out_channels=3, z_dim=20, samples=20, batch_size=32):
        super(decoder, self).__init__()
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.samples = samples
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.samples*self.z_dim, 1024)
        self.fc2 = nn.Linear(1024, 128*16*16)

        self.conv1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1)  #bs x 64 x 16 x 16
        self.conv2 = nn.Conv2d(64, 3, kernel_size=5, padding=2, stride=1) #bs x 3 x 16 x 16

        self.fc3 = nn.Linear(3*16*16, 3*32*32)
        self.fc4 = nn.Linear(3*32*32, 3*64*64)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.view(-1, self.samples*self.z_dim)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = x.view(self.batch_size, 128, 16, 16)  #bs x 128 x 16 x 16

        x = self.relu(self.conv1(x))
        #print(x.size())
        x = self.relu(self.conv2(x))
        #print(x.size())

        x = x.view(self.batch_size, 3*16*16)

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        #x = x.view(self.batch_size, 3, 64, 64)

        return x  #batchsize x 3*64*64