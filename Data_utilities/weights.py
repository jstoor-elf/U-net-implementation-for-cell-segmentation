# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:52:12 2019
@author: Hugo & Joakim
"""
import torch

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


"""
calculates d1, and then uses the resulting segmentation to
find distance to 2nd nearest cell
This probably does not scale well with more distances beyond d1, d2
This is about as fast as separation_weights2, depending on circumstance
"""

def separation_weights(mask, labels):
    labels=labels[labels > 0]
    # labels = labels[labels > 0]
    # the borders are created elsewhere, labeling pixels with d2=1 as 0
    #this should create a ridge between cells
    foreground = mask == 0
    d = np.zeros((3,) + mask.shape) #concatenate tuples
    d[0, :, :], inds1 = ndimage.distance_transform_edt(foreground, return_indices=True)
    seg2 = mask[[inds1[0, :, :], inds1[1, :, :]]]
    for label in labels:
        foreground = seg2 == label
        d[-1, :, :], inds_temp = ndimage.distance_transform_edt(foreground, return_indices=True)
        d[1, foreground] = d[0, inds_temp[0, foreground], inds_temp[1, foreground]] + d[-1, foreground]

    #two distance weight maps, d1 and d2
    return d[:-1, :, :]

"""
removes one cell one at a time, calculates distances and sorts them.
Can easily adapt to more neighbors
This is about as fast as separation_weights, depending on circumstance
does not have the expected weight output
"""
def separation_weights2(mask, labels, num_closest=2):
    labels=labels[labels > 0]
    # labels = labels[labels > 0]
    # the borders are created elsewhere, labeling pixels with d2=1 as 0
    #this should create a ridge between cells
    d = np.full((num_closest+1,)+ mask.shape, fill_value=np.inf) #initial distances, an extra layer for temporary
    for label in labels:
        foreground = (mask < 1) | (mask != label) #look at distances from one cell (label) at a time
        d[-1, :, :] = ndimage.distance_transform_edt(foreground)
        d = np.sort(d, axis=0)    #smallest distance at first layer and so on
    #two distance weight maps, d1 and d2
    return d[:-1, :, :]

"""
calculates weights for each pixel (balance and separation) and also
creates a border where two cells meet (resulting in a new mask)
returns weights, mask
"""
def weights_bordermask(mask,w0=10., sigma=5., v_bal=0.5, version=2):
    labels, inverse = np.unique(mask, return_inverse=True)

    if version==1: #more complicated, and slower ^^
        d = separation_weights(mask, labels) #change between function to see differences
    else:
        d = separation_weights2(mask, labels)

    labels = labels.astype(float)
    labels[labels > 0] = 1.
    labels[labels==0] = v_bal # assign new weights to the labels
    # w_bal[labels < 0 ] == 0 #unknown regions
    class_weights = np.reshape(labels[inverse], mask.shape)
    mask_with_border = np.copy(mask)
    mask_with_border[d[1, :, :] < 2], class_weights[d[1, :, :] < 2] = 0, v_bal
    d[:, mask_with_border > 0 ] = 3*sigma
    return w0*np.exp(-np.power(np.sum(d, axis=0),2)/(2*sigma**2)) + class_weights, mask_with_border


'''
The input is tensor masks, it may be on the gpu so we change device. Computes
the weight maps for the tiles. 
'''
def compute_weights(masks, bin_map=None, v_bal=0.5):
    masks = masks.to('cpu')
    for i in range(masks.size()[0]):
        weight_map, new_mask = weights_bordermask(masks[i].squeeze().numpy(), v_bal=v_bal)
        weight_map = torch.as_tensor(weight_map).float()
        new_mask = torch.as_tensor(new_mask).unsqueeze(0)

        if bin_map is not None:
            weight_map = (weight_map * bin_map[i, :, :])
        weight_map = (weight_map).unsqueeze(0)

        if i == 0:
            weight_maps = weight_map
            new_masks = new_mask
        else:
            weight_maps = torch.cat([weight_maps, weight_map], 0)
            new_masks = torch.cat([new_masks, new_mask], 0)

    return weight_maps.unsqueeze(1), new_masks > 0
