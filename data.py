import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
from PIL import Image

class dogDataset(Dataset):
    def __init__(self, image_dir, transformer1=None, transformer2=None):
        
        self.image_dir = image_dir
        self.image_types = os.listdir(image_dir)
        self.transformer1 = transformer1
        self.transformer2 = transformer2

        self.imgs = []
        for type in self.image_types:
            type_path = os.path.join(image_dir, type)
            for name in os.listdir(type_path):
                img = Image.open(os.path.join(type_path, name))

                if self.transformer1 is not None:
                    #image size : 64x64
                    img = self.transformer1(img)
                
                self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]

        if self.transformer2 is not None:
            img = self.transformer2(img)

        return img

    def __len__(self):
        return len(self.imgs)


