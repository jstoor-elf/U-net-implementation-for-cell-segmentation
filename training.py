import os

import torch
from  torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch import optim
from torchvision.transforms.functional import normalize, adjust_contrast, to_tensor
import torchvision.transforms as transforms
from torch.autograd import Variable

from Model import Unet
from Data_utilities.file_handler import GroundTruthFileHandler, get_abs_path
from Data_utilities.data_handler import CellDataset
from Data_utilities.visualize_data import dataset_viewer
from Data_utilities.augmentation_toolbox import generator, TrainingTransform
from Data_utilities.weights import compute_weights
from fitness_measures import evaluate, SoftLoss

import numpy as np
import numpy.random as random

'''
Main training regime using stochastic gradient descent, optionally with
momentum.
'''
def train_network(network, dataset, val_set=None, use_gpu=True, epochs=100, batch_size=1,
                  lr=0.05, mom=0.99, save_cp=True, net_save=20, val_save=10):


    # Device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu \
        else "cpu")
    network.to(device) # Put network on device


    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2, shuffle=False)
        f = open(get_abs_path() + "Networks/validation_set_scores.csv", "w+")
        f.write("Epoch,Cross Entropy Loss,Dice Loss\n")
        f.close()

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)


    print('''Start training:
    Epochs: {}
    Batch size: {}
    Optimizer: SGD
    Learning rate: {}
    Momentum: {}
    Training size: {}
    Validation size: {}
    Checkpoints: {}
    Device: {}
    '''.format(epochs, batch_size, lr, mom, len(dataset), 0 if val_set is None \
            else len(val_set), save_cp, str(device)))


    # Define network optimizer and criterion
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=mom, weight_decay=0.0005)
    criterion = SoftLoss() # Own criterion


    for epoch in range(epochs):
        network.train()
        loss = 0.0
        for i, sample in enumerate(dataloader):
            # Collect the image and its ground truth
            images, image_masks = sample['image'], sample['mask']

            # Generates sample for training, includes data augmentation
            tiles, masks, bin_map = generator(images, image_masks, nmbr_tiles=1)
            tiles, masks = tiles.to(device), masks.to(device)

            # Forward pass + max over the 2nd dimension to get predictions
            prob_mask = network(tiles)

            # Compute wight maps
            weight_map, new_masks = compute_weights(masks, bin_map, v_bal=1.5)
            weight_map, new_masks = weight_map.to(device), new_masks.to(device)

            # Batch loss
            crit = criterion.apply
            batch_loss = crit(prob_mask, new_masks, weight_map)

            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()


        print("Epoch {} is now finished".format(epoch + 1))


        # Alternatively save after every epoch
        if save_cp and (epoch % net_save == net_save-1):
            torch.save(network.state_dict(), get_abs_path() + "/Networks/" + \
                       "CP{}.pth".format(epoch + 1))
            print("Model after epoch {} saved!".format(epoch+1))


        if val_set is not None and (epoch % val_save == val_save-1):
            val_loss, val_dice = evaluate(network, val_loader, criterion)
            f = open(get_abs_path() + "Networks/validation_set_scores.csv", "a+")
            f.write("{},{},{}\n".format(epoch+1, val_loss, val_dice))
            f.close()


if __name__ == "__main__":

    load = False
    dataset = "DIC-C2DH-HeLa"

    n_channels, n_classes, f_depth = 1, 2, 64
    network = Unet(n_channels, n_classes, f_depth, ).float()

    if load:
        network.load_state_dict(torch.load(os.path.join(get_abs_path(), "Networks/CP2160.pth")))


    # Create dataset to be used for training
    tr_transform = TrainingTransform()
    filehandler_tr1 = GroundTruthFileHandler(dataset, film=1)
    filehandler_tr2 = GroundTruthFileHandler(dataset, film=2)
    dataset_tr1 = CellDataset(filehandler_tr1, tr_transform)
    dataset_tr2 = CellDataset(filehandler_tr2, tr_transform)
    tr_dataset = ConcatDataset((dataset_tr1, dataset_tr2))

    # We also create the validation set
    va_transform = TrainingTransform()
    filehandler_val = GroundTruthFileHandler(dataset, type='Validation')
    va_dataset = CellDataset(filehandler_val, va_transform)

    train_network(network, tr_dataset, val_set=tr_dataset, epochs=2400, net_save=24, val_save=6)
