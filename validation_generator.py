
import torch
from torchvision.transforms.functional import normalize, adjust_contrast, to_tensor
import numpy.random as random
from  torch.utils.data import DataLoader

from Data_utilities.file_handler import GroundTruthFileHandler, save_image
from Data_utilities.data_handler import CellDataset
from Data_utilities.visualize_data import dataset_viewer
from Data_utilities.augmentation_toolbox import rotate_and_deform, TrainingTransform
import matplotlib.pyplot as plt

'''
Generates more validation data than whats already given. Some images and their
corresponding masks have been taken out from the dataset and stored in files
for validation. These are then used to create more validation images using
both linear and non-linear transformations.
'''
def generate_data(dataset, transform=None):

    filehandler = GroundTruthFileHandler(dataset, type='Validation')
    val_set = CellDataset(filehandler, transform)
    dataloader = DataLoader(val_set, batch_size=1, num_workers=2, shuffle=False)

    cnt = 2
    while cnt < 16 + len(val_set):
        for i, sample in enumerate(dataloader):
            images, image_masks = sample['image'], sample['mask']

            image, mask = rotate_and_deform(images, image_masks, sigma=random.randint(15,25))
            save_image(image.squeeze(), dataset, name='t' + str(cnt).zfill(3) + '.tif')
            save_image(mask.squeeze(), dataset, type="mask", \
                name='man_seg' + str(cnt).zfill(3) + '.tif')
            cnt += 1


''' Shows the result of the generated data, should be checked before training '''
def show_genrated_data(dataset):
    filehandler = GroundTruthFileHandler(dataset, type='Validation')
    val_set = CellDataset(filehandler, transform)
    dataset_viewer(val_set, True)


''' Here we can generate the data and show it '''
if __name__ == "__main__":
    dataset = "DIC-C2DH-HeLa"

    transform = TrainingTransform()
    # generate_data(dataset, transform=transform) # Data already generated
    show_genrated_data(dataset)
