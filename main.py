#created by liudan 
#2023/6/6

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import optim
import os
import argparse

from model.vae import vae
from data import dogDataset
from loss import vae_loss

def train(model, train_loader, optimizer, epoch, batch_size):
    model.train()

    train_loss = 0
    count = 0

    for batch_id, data in enumerate(train_loader):
        #print(data.size())
        x = data
        count += x.size(0)
        x = Variable(x.type(torch.FloatTensor).cuda())
        
        optimizer.zero_grad()
        x_new, z_mu, z_sigma = model(x)
        #x_new = x_new.detach()
        loss = vae_loss(x, x_new, z_mu, z_sigma, batch_size)
        train_loss += loss

        loss.backward()
        optimizer.step()

        if batch_id % 50 ==0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss))
            
        train_loss /= count
        print('\nTrain set: Average loss: {:.4f}'.format(train_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="epoch of training")
    parser.add_argument("--samples", type=int, default=20, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=8, help="set the batch_size for training")
    parser.add_argument("--gpu", type=str, help="gpu", default="3")

    arg = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer1 = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64)])

    # Data augmentation and converting to tensors
    random_transforms = [transforms.RandomRotation(degrees=10)]
    transformer2 = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply(random_transforms, p=0.3), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = dogDataset('/scratch/liudan/personal/dogs_dataset/Images', transformer1, transformer2)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=arg.batch_size,
                            shuffle=True,
                            num_workers=4)
    print("succesfully load data!")

    z_dim = 20
    model = vae(z_dim, arg.samples, arg.batch_size)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=.001)

    print("Start training!")
    for epoch_i in range(arg.epoch):
        train(model, train_loader, optimizer, epoch_i+1, arg.batch_size)

if __name__=="__main__":
    main()


