###################################################################################################
# MemeNet dataloader
# Marco Giordano
# Center for Project Based Learning
# 2022 - ETH Zurich
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

    def __init__(self, data_dir=None, transform=None, filename="ims_bearings_train_exp1_b3_spectrograms"):
        self.data_dir = data_dir
        self.transform = transform
        self.filename = filename
        self.samples = torch.load(os.path.join(self.data_dir, "%s"%self.filename))
        #shuffle samples
        torch.manual_seed(0)
        ra = torch.randperm(self.samples.shape[0])
        self.samples = self.samples[ra]
        
        self.samples_len = self.samples.shape[0]
        
    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx):

        spectrogram = self.samples[idx]
        label = spectrogram
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            label = spectrogram
            
        return spectrogram, label

"""
Dataloader function
"""
def ims_bearings_get_datasets(data, load_train=False, load_test=False):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = IMSBearingsDataset(data_dir=os.path.join(data_dir, "ims_bearings", "train"), transform=train_transform, filename="ims_bearings_train_exp1_b3_spectrograms.pt")
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = IMSBearingsDataset(data_dir=os.path.join(data_dir, "ims_bearings", "test"), transform=test_transform, filename="ims_bearings_test_exp1_b3_spectrograms.pt")

    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'ims_bearings',
        'input': (1, 64, 64),
        'output': (1, 64, 64),
        'loader': ims_bearings_get_datasets,
    }
]



# if __name__ == '__main__':
#     # dataset, _ = memes_get_datasets("./data/memes/train/", True)
#     dataloader = DataLoader(memes_get_datasets("./data", load_train=False, load_test=True), batch_size=4,
#                         shuffle=True, num_workers=0)

#     fig, ax = plt.subplots(4, 4)

#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch, sample_batched[0].size(),
#             sample_batched[1].size())

#         # observe 4th batch and stop.
#         if i_batch < 4:
#             for i, img in enumerate(sample_batched[0]):
#                 print(img.shape)
#                 ax[i_batch, i].imshow(img.permute((1,2,0)))
                
#     plt.title('Batch from dataloader')
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
