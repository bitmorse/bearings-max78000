###################################################################################################
# BearingNet dataloader
# Sam Sulaimanov
# 2023
###################################################################################################
"""
MemeNet dataset
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_image

import ai8x
import torch

import os
import pandas as pd
import time

import matplotlib.pyplot as plt
from PIL import Image

"""
Custom image dataset class
"""
class IMSBearingsDataset(Dataset):

    def __init__(self, data_dir=None, full=False, transform=None, filename="ims_bearings_train_exp1_b3_spectrograms"):
        self.data_dir = data_dir
        self.transform = transform
        self.filename = filename
        self.samples = torch.load(os.path.join(self.data_dir, "%s"%self.filename))
        #shuffle samples
        #torch.manual_seed(0)
        #ra = torch.randperm(self.samples.shape[0])
        #self.samples = self.samples[ra]
        
        #self.samples_len = self.samples.shape[0]
        if full:
            self.samples_len = self.samples.shape[0]
        else:
            self.samples_len = self.samples.shape[0]//5 #artificial limit by me!
        
    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx):
        spectrogram = self.samples[idx]
        
        assert spectrogram.shape[0] == 50
        assert spectrogram.shape[1] == 50
        
        #cut off 1 px board of spectrogram
        label = spectrogram
        
        assert label.shape[0] == 50
        assert label.shape[1] == 50
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            label =  self.transform(spectrogram)
            
        return spectrogram, label

"""
Dataloader function
"""
def ims_bearings_get_datasets(data, load_train=False, load_test=False, full=False):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = IMSBearingsDataset(data_dir=os.path.join(data_dir, "ims_bearings", "train"), transform=train_transform, filename="ims_bearings_train_exp1_b3_spectrograms.pt", full=full)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(), 
            ai8x.normalize(args=args)
        ])

        test_dataset = IMSBearingsDataset(data_dir=os.path.join(data_dir, "ims_bearings", "test"), transform=test_transform, filename="ims_bearings_test_exp1_b3_spectrograms.pt", full=full)

    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'ims_bearings',
        'input': (1, 50, 50),
        'output': (1, 50, 50),
        'loader': ims_bearings_get_datasets,
    }
]