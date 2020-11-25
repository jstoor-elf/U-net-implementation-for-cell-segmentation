# Mainly for input
import os
from os import listdir
from os.path import isfile, join

# Regular expression used to get files
import re
from PIL import Image
import torchvision.transforms.functional as F


from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np

NOT_GOOGLE_CLOUD = True # Used to set absolute path


""" For training and validation datasets """
class GroundTruthFileHandler():
    def __init__(self, dataset, film=1, type='Training'):
        self.dataset = dataset
        self.im_dir = get_path_to_data_set(dataset, 'image', film, type)
        self.mask_dir = get_path_to_data_set(dataset, 'mask', film, type)
        self.image_files, self.mask_files = get_file_names(self.mask_dir, self.im_dir)


""" For testing datasets without ground truths """
class CleanFileHandler():
    def __init__(self, dataset, film=1):
        self.dataset = dataset
        self.im_dir = get_path_to_data_set(dataset, 'image', film, d_type="Testing")
        self.image_files = get_all_files(self.im_dir)


"""
type can be 'mask' or 'image', film can be 1 or 2
d_type can be Training or Testing
"""
def get_path_to_data_set(data_set, type, film=1, d_type="Training"):
    path = os.path.join(get_abs_path(), "Data_utilities/Data", d_type, data_set)
    numb = "01" if film == 1 else "02"
    if d_type == "Testing":
        ext = numb if type == "image" else os.path.join(numb + "_RES")
    else:
        ext = numb if type == "image" else os.path.join(numb + "_GT/SEG")
    return os.path.join(path, ext)


"""
For training datasets that involves all the images, no masks
"""
def get_all_files(im_dir):
    file_names = sorted(store_filenames_from_dir(im_dir, 'tif'))
    return {key : file for key, file in enumerate(file_names)}


""" Returns image files and image masks of interest, i.e. for given dataset """
def get_file_names(mask_dir, im_dir):
    mask_file_names = sorted(store_filenames_from_dir(mask_dir, 'tif'))
    file_names = sorted(store_filenames_from_dir(im_dir, 'tif'))

    # Manual tests, looks akward but done for validation
    if mask_dir == get_abs_path() + "Data_utilities/Data/Training/DIC-C2DH-HeLa/01_GT/SEG":
        if "man_seg067.tif" in mask_file_names:
            mask_file_names.remove("man_seg067.tif")
        if "man_seg002.tif" in mask_file_names:
            mask_file_names.remove("man_seg002.tif")

    if mask_dir == get_abs_path() + "Data_utilities/Data/Training/DIC-C2DH-HeLa/02_GT/SEG":
        if "man_seg006.tif" in mask_file_names:
            mask_file_names.remove("man_seg006.tif")


    numbers = []
    for mask_file_name in mask_file_names:
        numbers.append(re.findall('\d+',mask_file_name)[-1])

    image_file_names = []
    for file in file_names:
        number = re.findall('\d+',file)[-1]
        if number == numbers[len(image_file_names)]:
            image_file_names.append(file)
        if (len(image_file_names) == len(numbers)):
            break

    res1 = {key : file for key, file in enumerate(image_file_names)}
    res2 = {key : file for key, file in enumerate(mask_file_names)}

    return res1, res2

'''
Can be used to save validation images and masks, but primarily used
to save segmentation masks for the testing images
'''
def save_image(image, dataset, name, type='image', film=1, d_type="Validation"):
    path = os.path.join(get_abs_path(), "Data_utilities/Data", d_type, dataset)
    numb = "01" if film == 1 else "02"
    if d_type == "Validation":
        ext = numb if type == "image" else numb + "_GT/SEG"
    else:
        ext = numb if type == "image" else numb + "_RES"
    full_name = os.path.join(path, ext, name)
    if type == "image":
        image = image * 255 # Get rid of transform
    pil_im = F.to_pil_image(image)
    pil_im.save(full_name)



"""
Picks all the file with extension ext in directory goven from path_name
"""
def store_filenames_from_dir(path_name, ext):
    return [f for f in listdir(path_name) if\
        isfile(join(path_name, f)) and f.endswith(".{}".format(ext))]


''' Working directory '''
def get_abs_path():
    if NOT_GOOGLE_CLOUD:
        return "" # Working directory
    else:
        return "" # Working directory on google cloud
