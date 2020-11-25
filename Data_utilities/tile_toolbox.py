
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
import numpy as np
from random import sample

from PIL import Image

limit = 172 # Used to find valid shape, empirically found


''' Takes in a tile input shape and returns if its valid or not '''
def valid_input_size(tile_shape):
    # This equation was found empirically, 172 + n*16 = tile_shape
    diff = ((np.array(tile_shape) - limit) // 16)
    if (diff > 0).all():
        if ((np.array(tile_shape) - limit) / 16 == diff).all():
            return True
    return False


''' Returns valid input tile shapes within the range [start_value, end_value] '''
def find_valid_input_tile_shapes(start_value, end_value):
    potential_sizes = []

    start_value = limit if start_value <= limit else start_value + int(start_value % 2)
    for i in range(start_value, end_value, 2):
        if (i - limit) / 16 == (i - limit) // 16 and (i - limit) // 16 > 0 :
            potential_sizes.append(i)

    return potential_sizes


'''
Takes in a 2D tuple representing the output tile shape and returns the
corresponding input tile shape, works equally well for the reverse case
'''
def find_shape(shape, upscale=True):
    levels = 4 # Number of levels, network specific
    op = operator.add if upscale == True else operator.sub
    shape = op(shape, np.sum([2**x for x in range(2, levels+3)])) // (2**levels)
    shape = op(shape * (2**levels), np.sum([2**x for x in range(2, levels+2)]))
    return tuple(shape)



'''
Get tile coordinates, list of dicts for x_start, x_end, y_start, y_end based
on im_shape and input_tile
'''
def get_tile_coordinates(im_shape, input_tile):
    output_tile = find_shape(input_tile, upscale=False)
    padding = (np.array(input_tile) - np.array(output_tile) ) // 2
    new_im_shape = np.array(im_shape) + padding * 2

    # Gives number of turns in x and y direction
    looper = np.ceil(np.array(im_shape) / output_tile).astype(int)
    tile_coordinates = [] # Tile coordinates

    y = 0
    for i in range(looper[0]):
        x, y_e = 0, y + padding[0] * 2 + output_tile[0]
        for j in range (looper[1]):
            x_e = x + padding[1] * 2 + output_tile[1]
            tile_coordinates.append({'x_s' : x, 'x_e' : x_e, 'y_s' : y, 'y_e' : y_e})
            x += output_tile[1] # Increment x

        y += output_tile[0] # Increment y

    return tile_coordinates, padding, looper



''' Randomly Generate tiles in image, the number of tile can be specified  '''
def tile_generator(image, tile_size, nmbr_tiles=4):
    limit = image.size()[2:] - tile_size
    y_index = sample(range(0,limit[0]), nmbr_tiles)
    x_index = sample(range(0,limit[1]), nmbr_tiles)

    output_tiles_size = np.array(find_shape(tile_size, upscale=False))
    diff = (tile_size - output_tiles_size) // 2

    coords_input, coords_output  = [], []
    for x, y in zip(x_index, y_index):
        coords_input.append({'xs' : x, 'xe' : x + tile_size[1], \
                       'ys' : y, 'ye' : y + tile_size[0]})

        coords_output.append({'xs' : x + diff[1], 'xe' : x + tile_size[1] \
            - diff[1], 'ys' : y + diff[0], 'ye' : y + tile_size[0] - diff[0]})

    return coords_input, coords_output


'''
Takes in a tensor and padding and returns a rescaled padded tensor that has
been reflected at the borders
'''
def pad_tensor(im_tensor, padding, tile_size):

    # Get tile and image sizes
    tile_size = np.asarray(find_shape(tile_size, upscale=False))
    image_size = im_tensor.size()[2:]

    # Find correct padding as function of the net architecture and input tile
    cs = (np.ceil(image_size / tile_size) * tile_size - image_size).astype(int)
    padding = np.array(padding) + (cs // 2)
    pads = (padding[0], padding[0], padding[1], padding[1])

    return F.pad(im_tensor, pads, mode='reflect')



'''
Returns the padding that's used to crop out the central part of final the
segmentation
'''
def get_last_bit(input_tile, im_size, looper):
    output_tile = find_shape(input_tile, upscale=False)
    return output_tile * looper - im_size
