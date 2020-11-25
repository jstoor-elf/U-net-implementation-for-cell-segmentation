import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, adjust_contrast, to_tensor
from  torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from prediction import predict_image
from Data_utilities.file_handler import GroundTruthFileHandler, get_abs_path
from Data_utilities.file_handler import CleanFileHandler, save_image
from Data_utilities.data_handler import CellDataset, TestingDataset
from Data_utilities.augmentation_toolbox import TrainingTransform, TestingTransform
from Data_utilities.tile_toolbox import find_shape, valid_input_size
from Data_utilities.visualize_data import dataset_viewer
from Model import Unet
from fitness_measures import dice_measure, pixel_accuracy, mean_accuracy, jacard_score

from skimage import measure


''' Plots the validation test results from training. '''
def validate_model():

    variables_file =  get_abs_path() + 'Networks/validation_set_scores.csv'

    # Read the CSV into a pandas data frame (df)
    df = pd.read_csv(variables_file, header = 0, delimiter=',')

    #fig, axes = plt.subplots(nrows=1, ncols=2, dpi=80)
    df.plot(kind='line',x='Epoch',y=['Cross Entropy Loss', 'Dice Loss'], \
        color=['brown','orange'], style=['-','-'])

    l = np.argmin(df['Cross Entropy Loss'])
    print('Optimal model found at epoch {} with cross entropy loss {}'\
        .format(df['Epoch'][l],np.min(df['Cross Entropy Loss'])))

    plt.show()


def save_and_store_test_data(network, dataset, dataset_name, film=1):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    for index, sample in enumerate(dataloader):
        print("Saving image {} ... ".format(index+1), end='')
        image, im_file = sample
        _, pred_mask = predict_image(network, image, input_tile=(716, 716))

        labeled = measure.label(pred_mask.squeeze().squeeze().numpy(), background=0)
        pred_mask = torch.from_numpy(labeled).unsqueeze(0).short()

        save_image(pred_mask.short(), dataset_name, name=im_file[0], type='mask',
            film=film, d_type="Testing")

        print("OK")


def plot_prob_map(network, dataset, idx):

    idx = idx % len(dataset)
    image, mask = dataset[idx]['image'].unsqueeze(0), dataset[idx]['mask']
    final_tensor, pred_mask = predict_image(network, image)

    # Apply softmax on the final tensor from the evaluation to get a probability map
    final_tensor = F.softmax(final_tensor, dim=1)

    # Labeling connected components for cell detection
    blobs_labels = measure.label(pred_mask.squeeze().squeeze().numpy(), background=0)

    plt.figure(figsize=(13, 2.1))
    plt.subplot(151)
    plt.imshow(image.squeeze().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(152)
    plt.imshow(mask.squeeze(), cmap="hot")
    plt.axis('off')
    plt.subplot(153)
    plt.imshow(pred_mask.squeeze(), cmap="hot")
    plt.axis('off')
    plt.subplot(154)
    plt.imshow(blobs_labels, cmap="hot")
    plt.axis('off')
    plt.subplot(155)
    plt.imshow(final_tensor[:,1,:,:].squeeze(), cmap="BrBG")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.show()


def validate_scores(network, dataset):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    results = np.zeros(shape=(1,4))
    cols = ['pixel accuracy', 'mean_accuracy', 'dice score', 'jacard score']
    for index, sample in enumerate(dataloader):
        ('Processing image {}'.format(index+1))
        image, image_mask = sample['image'], sample['mask'] > 0
        _, pred_mask = predict_image(network, image)

        results[0,0] += pixel_accuracy(pred_mask, image_mask)
        results[0,1] += mean_accuracy(pred_mask, image_mask)
        results[0,2] += (-1)*(dice_measure(pred_mask, image_mask) - 1)
        results[0,3] += jacard_score(pred_mask, image_mask)


    results /= len(dataset)
    results_df = pd.DataFrame(data=results, index=['o'], columns=cols)
    results_df.plot(kind='bar')
    print(results_df.head())


# if __name__ == "__main__": Implement for arbitrary test
