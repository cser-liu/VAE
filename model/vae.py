import torch
import torch.nn as nn
from torch.autograd import Variable

from model.encoder import encoder
from model.decoder import decoder

class vae(nn.Module):
    """
    todo
    """

    def __init__(self, z_dim=20, samples=20):
        super(vae, self).__init__()

        self.z_dim = z_dim
        self.samples = samples
        #self.batch_size = batch_size

        self.vae_encoder = encoder(3, self.z_dim)
        self.vae_decoder = decoder(3, self.z_dim, self.samples)

    def forward(self, x):
        z_mu, z_sigma = self.vae_encoder(x) #bs x z_dim

        #re-parameterize
        bs = z_mu.size(0)
        standard_sample = Variable(torch.randn(bs, self.samples,self.z_dim)) #bs x samples x z_dim
        standard_sample = standard_sample.cuda()
        z_mu = z_mu.unsqueeze(1)
        z_sigma = torch.exp(0.5 * z_sigma)
        z_sigma = z_sigma.unsqueeze(1)
        z = z_mu + z_sigma*standard_sample #bs x samples x z_dim

        x_new = self.vae_decoder(z)

        return x_new, z_mu, z_sigma

