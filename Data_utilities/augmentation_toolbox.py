
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize, adjust_contrast, to_tensor

import elasticdeform
import numpy as np
import numpy.random as random
from random import sample

from .tile_toolbox import find_valid_input_tile_shapes, find_shape
from .tile_toolbox import tile_generator, pad_tensor

'''
This generator generates tiles from  image and achieves translation, rotation,
scaling, shering, and non-linear deformation, as it is implemented now
batch_size for input tensor image has to be of size 1
'''
def generator(image, mask, nmbr_tiles=1, sigma=20):
    # Rotation and non-linear deformation, hopefully with shearing
    image, mask = rotate_and_deform(image, mask, sigma=sigma)

    # Randomize crop size for scaling
    tile_size = random_crop_size()

    # Find output tile size for given input tile size and pad image
    init_image_size = image.size()[2:]
    image, mask = pad_image(image, mask, tile_size)

    # Create binary map
    binary_map = create_binary_map(image.size()[2:], init_image_size)

    # Randomly generate tiles of size tile_size --> translation
    coords_input, coords_output = tile_generator(image, tile_size, nmbr_tiles=1)
    image_tiles, mask_tiles, ba_tile =  create_tiles(image, mask, \
        coords_input, coords_output, binary_map)


    return image_tiles, mask_tiles, ba_tile


'''
From an image size and a padded image size, creates a binary map which is
multiplied with the weight map to give zero weighting to padded regions
'''
def create_binary_map(out_size, in_size):
    binary_map = torch.zeros(out_size)
    padding = (np.array(out_size) - np.array(in_size)) // 2
    binary_map[padding[0]:-padding[0], padding[1]:-padding[1]] = 1

    return binary_map.unsqueeze(0)


''' Takes in an input tile and rotates and deforms it '''
def rotate_and_deform(image, mask, sigma=25):
    axes = sample(range(2,4), random.randint(0,3))

    # Rotate image and deform
    rot_im, rot_ma = torch.flip(image, dims=axes), torch.flip(mask, dims=axes)
    rot_im, rot_ma = rot_im.numpy(), rot_ma.numpy()
    [im_def, mask_def]  = elasticdeform.deform_random_grid([rot_im, rot_ma], axis=(2,3), \
        sigma=sigma, mode='mirror', prefilter=False, order = [3, 0])

    return torch.as_tensor(im_def), torch.as_tensor(mask_def.round())


'''
Creates the tiles that are returned by the generator to be used during
training. Takes in coordinates that defines the image window to create the
tiles from.
'''
def create_tiles(image, mask, coords_input, coords_output, binary_map):
    ''' Loops through coordinates and generates tiles '''

    for i in range(len(coords_input)):
        im = image[:,:, coords_input[i]['ys'] : coords_input[i]['ye'], \
            coords_input[i]['xs'] : coords_input[i]['xe']]
        ma = mask[:,:, coords_output[i]['ys'] : coords_output[i]['ye'], \
            coords_output[i]['xs'] : coords_output[i]['xe']]
        ba = binary_map[:, coords_output[i]['ys'] : coords_output[i]['ye'], \
            coords_output[i]['xs'] : coords_output[i]['xe']]

        if i == 0:
            im_tile, ma_tile, ba_tile = im, ma, ba

        else:
            im_tile = torch.cat([im_tile, im], 0)
            ma_tile = torch.cat([ma_tile, ma], 0)
            ba_tile = torch.cat([ba_tile, ba], 0)


    return im_tile, ma_tile, ba_tile


''' Selects a ranom input tile size within the specified interval '''
def random_crop_size(start_index=380, end_index=600):
    input_sizes = find_valid_input_tile_shapes(start_index, end_index)
    index = random.randint(0, len(input_sizes))
    tile_size = np.array([input_sizes[index], input_sizes[index]])
    return tile_size


'''
Pads an image. The amount of padding is a product of the network architecture
and the input tile size
'''
def pad_image(image, mask, tile_size):
    output_tile = find_shape(tile_size, upscale=False)
    padding = (tile_size - np.array(output_tile) ) // 2
    image = pad_tensor(image, padding, tile_size)
    mask = pad_tensor(mask.float(), padding, tile_size).int()

    return image, mask


''' Transform for training and validation data '''
class TrainingTransform(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = to_tensor(image).float()
        image = normalize(image, mean=(0,), std=(255,)) #sets equal contrast to images
        mask = to_tensor(mask)

        return {'image' : image, 'mask' : mask}



''' Transform for test data '''
class TestingTransform(object):

    def __call__(self, image):
        image = to_tensor(image).float()
        image = normalize(image, mean=(0,), std=(255,)) #sets equal contrast to images

        return image
