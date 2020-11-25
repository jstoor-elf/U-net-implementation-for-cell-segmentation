import os

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

import numpy as np
import Data_utilities.tile_toolbox as tt
from Data_utilities.file_handler import GroundTruthFileHandler, CleanFileHandler, get_abs_path
from Data_utilities.data_handler import CellDataset, TestingDataset
from Data_utilities.visualize_data import dataset_viewer
from Data_utilities.augmentation_toolbox import generator


datasets = {
    1 : "DIC-C2DH-HeLa", # Good
    2 : "PhC-C2DL-PSC", # Good
    3 : "PhC-C2DH-U373", # Good
    4 : "Fluo-N2DH-SIM+", # Ok, lots of masks
    5 : "Fluo-C2DL-MSC", # Less good
    6 : "Fluo-N2DH-GOWT1", # Not good
    7 : "Fluo-N2DH-SIM+", # Ok, lots of masks
}

def predict_image(network, image, use_gpu=True, input_tile=(476, 476),
                  save_file_with_name=None):

    # First do a check on the input tile, check if its valid or not
    if ~ tt.valid_input_size(input_tile):
        Exception('Incorrect input tile size')

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    im_size = image.size()[2:] # Get e.g. 512*512
    coordinates, padding, looper = tt.get_tile_coordinates(im_size, input_tile)

    # Create image tensor with padding
    im_tensor = tt.pad_tensor(image, padding, input_tile)


    # Create tiles of interest
    for cnt, coords in enumerate(coordinates):
        if cnt == 0:
            tiles = im_tensor[:, :, coords['y_s']:coords['y_e'], coords['x_s']: \
                coords['x_e']]
        else:
            tiles = torch.cat([tiles, im_tensor[:, :, coords['y_s']:coords['y_e'],\
             coords['x_s']: coords['x_e']]], 0)


    tiles.to(device) # Put tensor on the device of choice

    batch_size = tiles.size()[0] // (looper[0] * looper[1])

    with torch.no_grad():
        vec, cnt = [], 0
        for y in range(looper[0]):
            for x in range(looper[1]):
                indexes = range(cnt*batch_size, (cnt+1)*batch_size)
                tile = tiles[indexes, :, :,:].to(device)

                prob_map = network(tile)

                if x == 0:
                    prob_maps = prob_map
                else:
                    prob_maps = torch.cat([prob_maps, prob_map], 3)

                cnt += 1
            vec.append(prob_maps)


        # Indecies used to crop out the image
        ind = tt.get_last_bit(input_tile, im_size, looper) // 2

        final_tensor = torch.cat(vec, 2)
        final_tensor = final_tensor[:, :, ind[0] : - ind[0], ind[1] : - ind[1]]

    # Just for plotting at the moment, will be changed later
    _, pred_mask = final_tensor.max(1)

    return final_tensor, pred_mask
