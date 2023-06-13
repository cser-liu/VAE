#created by liudan 
#2023/6/6

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torch import optim
import os
import argparse

from model.vae import vae
from data import dogDataset
from loss import vae_loss
from visualize import showImage
        

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
                                    transforms.ToTensor()
                                    ])

    dog_dataset = dogDataset('/scratch/liudan/personal/dogs_dataset/Images', transformer1, transformer2)
    train_size = int(0.8*len(dog_dataset))
    val_size = len(dog_dataset) - train_size
    train_dataset, val_dataset = random_split(dog_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=arg.batch_size,
                            shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=arg.batch_size,
                            shuffle=True,
                            num_workers=4)
    print('train_dataset has {} images, val_dataset has {} images.'.format(len(train_dataset), len(val_dataset)))
    #showImage(train_dataset[0].permute(1,2,0))
    #print("succesfully load data!")

    z_dim = 128
    model = vae(z_dim, arg.samples)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=.001)

    print("Start training!")

    train_losses = []
    valid_losses = []

    best_loss = 1e9
    best_epoch = 0

    for epoch_i in range(arg.epoch):
        print(f"Epoch {epoch_i}")
        model.train()

        train_loss = 0
        #count = 0

        for idx, x in enumerate(train_loader):
            #print(x.size())
            batch = x.size(0)
            #count += batch
            x = Variable(x.type(torch.FloatTensor).cuda())

            x_new, z_mu, z_sigma = model(x)

            batch_size = z_mu.size(0)
            BCE, KLD = vae_loss(x, x_new, z_mu, z_sigma, batch_size)
            loss = BCE+KLD
            train_loss += loss.item()
            loss = loss/batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 ==0:
                print(f"Training loss {loss: .3f} \t Recon {BCE/batch: .3f} \t KL {KLD/batch: .3f} in Step {idx}")
            
        train_losses.append(train_loss/len(train_dataset))

        valid_loss = 0
    
        model.eval()
        with torch.no_grad():
            for idx, x in enumerate(val_loader):
                x = Variable(x.type(torch.FloatTensor).cuda())
                x_new, z_mu, z_sigma = model(x)

                batch_size = z_mu.size(0)
                BCE, KLD = vae_loss(x, x_new, z_mu, z_sigma, batch_size)
                loss = BCE+KLD
                valid_loss += loss.item()

            valid_losses.append(valid_loss/len(val_dataset))
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch_i

                torch.save(model.state_dict(), 'output/new/best_model')
                print("Model saved")

    print("Training complete!")

if __name__=="__main__":
    main()


