import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 128
image_size = 64

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


#%% kendimize özel data set yükleme classı yazalım
class CustomImageDataset(Dataset): 
    def __init__(self, folder_path, transform=None):
        self.files = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

dataset = CustomImageDataset("train/images", transform=transform)

# veri setinin batchler halinde yukle
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%% Discriminator olusturalım

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace= True),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 4, stride=1, padding=0),
            nn.Sigmoid()
            
            
        )
        
    def forward(self,x):
        return self.model(x)



#%% generator modeli 

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # girdi: (z_dim,) → reshape ile (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),    # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),      # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),      # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),       # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),         # (3, 64, 64)
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)  # (batch, z_dim) → (batch, z_dim, 1, 1)
        return self.model(x)

            
            
            
            
            
            
            
            
            
            
            )

































