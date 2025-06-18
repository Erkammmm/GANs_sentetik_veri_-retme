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








































