from model.vae import vae
import torch
import torch.nn.functional as F

def vae_loss(x, x_new, z_mu, z_sigma, batch_size):
    x = x.view(-1, 3*64*64)

    #x_new = x_new.clamp(0, 1)
    #print(x[0, :100])
    #print(x_new[0, :100])
    BCE = F.binary_cross_entropy(x_new, x)
    #print(BCE)

    KLD = -0.5 * torch.sum(1 + z_sigma - z_mu.pow(2) - z_sigma.exp())
    KLD /= 3 * 64 * 64
    #print(KLD)
    return BCE, KLD