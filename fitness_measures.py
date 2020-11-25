import torch

from PIL import Image

from prediction import predict_image
from torch.autograd import Function, Variable

from Data_utilities.weights import compute_weights
from Data_utilities.augmentation_toolbox import create_binary_map
import matplotlib.pyplot as plt
import torch.nn.functional as F


"""
Loss function used for training and validation. Applies softmax and pixel wise
weighted cross entropy.
"""
class SoftLoss(Function):

    @staticmethod
    def forward(ctx, input, target, weight_map):

        # Create one hot encoding
        one_hot = torch.zeros_like(input)
        for x in range(2):
            one_hot[:,x,:,:] = (target == x).squeeze(1)

        # apply softmax for probabilities
        p = F.softmax(input, dim=1) #softmax along class
        #plot(p)
        w_sum = weight_map.sum() #calculate before we double, to math the number of 1s in onehot

        # 2x Concatenated weight map for efficient calculations in criterion
        weight_map = weight_map.repeat(1,2,1,1)/w_sum

        # probability distribution input
        y = one_hot.float()*weight_map

        ctx.save_for_backward(p, y, weight_map)
        return -torch.dot(y.view(-1), torch.log(p.view(-1))) #elementwise sum


    @staticmethod
    def backward(ctx, gradient_output):
        # dL/doi = -yi (1-pi) + sum[k not i](yk*pi)
        p, y, weight_map = ctx.saved_tensors # to get aligned with p
        grad = p*weight_map - y

        return gradient_output*grad, None, None #pi * weight(pixel) - y_i,


"""
Evaluates the network with dice loss score and pixel wise weighted cross
entropy loss
"""
def evaluate(network, dataloader, criterion, use_gpu=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    loss_tot, dice_tot, i = 0.0, 0.0, len(dataloader)
    with torch.no_grad():
        for index, sample in enumerate(dataloader):

            image, true_mask = sample['image'], sample['mask']

            prob_tensor, pred_mask = predict_image(network, image, use_gpu=use_gpu, \
                                    input_tile = (476, 476))

            prob_tensor, pred_mask = prob_tensor.to(device), pred_mask.to(device)

            weight_map, new_mask = compute_weights(true_mask, v_bal=1.5)
            weight_map, new_mask = weight_map.to(device), new_mask.to(device)

            crit = criterion.apply
            loss = crit(prob_tensor, new_mask, weight_map)
            dice = dice_measure(pred_mask, new_mask)
            loss_tot, dice_tot = loss_tot + loss, dice_tot + dice


    return loss_tot/i, dice_tot/i



''' Dice loss score for evaluation, is highly sensitive to initialization '''
def dice_measure(prob_mask, true_mask):

    smooth = 1.
    intersection = (prob_mask.float()*true_mask.float()).sum()
    denom = prob_mask.pow(2).sum() + true_mask.pow(2).sum()

    return 1 - 2*(intersection + 1) / (denom + 1)


''' Implementation for binary segmentation '''
def pixel_accuracy(prediction, ground_truth):
    prediction, ground_truth = prediction.float(), ground_truth.float()

    num_right = torch.sum(prediction*ground_truth) + torch.sum((1-prediction)*(1-ground_truth))
    num_pixels = torch.numel(ground_truth)
    return num_right/num_pixels


''' Implementation for binary segmentation '''
def mean_accuracy(prediction, ground_truth):
    prediction, ground_truth = prediction.float(), ground_truth.float()

    m0 = torch.sum((1-prediction)*(1-ground_truth))/(torch.numel(ground_truth)-torch.sum(ground_truth))
    m1 = torch.sum(prediction*ground_truth)/torch.sum(ground_truth)
    return (m0+m1)/2


''' Implementation for binary segmentation '''
def jacard_score(prediction, ground_truth):
    prediction, ground_truth = prediction.float(), ground_truth.float()

    intersection = (prediction*ground_truth).sum()
    union = ground_truth.sum() + ground_truth.sum() - intersection

    return intersection / union
