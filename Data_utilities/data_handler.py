# Mainly for input
import os
from os.path import join

from PIL import Image
from  torch.utils.data import Dataset
import numpy as np


datasets = {
    1 : "DIC-C2DH-HeLa", # Good
    2 : "PhC-C2DL-PSC", # Good
    3 : "PhC-C2DH-U373", # Good
    4 : "Fluo-N2DH-SIM+", # Ok, lots of masks
    5 : "Fluo-C2DL-MSC", # Less good
    6 : "Fluo-N2DH-GOWT1", # Not good
    7 : "Fluo-N2DH-SIM+", # Ok, lots of masks
}

'''
Dataset class used for the training and validation datasets which contains
ground truth segmentation masks
'''
class CellDataset(Dataset):
    def __init__(self, data_set_file_handler, transform=None):
        self.handler = data_set_file_handler
        self.transform = transform

    def __len__(self):
        return len(self.handler.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.handler.im_dir, self.handler.image_files[idx])
        mask_name = os.path.join(self.handler.mask_dir, self.handler.mask_files[idx])
        image, mask = Image.open(img_name), Image.open(mask_name)
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
Dataset class used for the testing dataset which doesn't contain ground
truths, used to generate image masks in an efficient way. When these masks have
been created the normal CellDataset can be used to visualize the segmentation
results.
'''
class TestingDataset(Dataset):
    def __init__(self, data_set_file_handler, transform=None):
        self.handler = data_set_file_handler
        self.transform = transform

    def __len__(self):
        return len(self.handler.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.handler.im_dir, self.handler.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.handler.image_files[idx]
