# U-net implementation for semantic cell segmentation
Mainly a PyTorch reproduction of the U-net design that was introduced by the authors of the now famous [U-net paper](https://arxiv.org/abs/1505.04597). The model was trained on images from the 'DIC-C2DH-HeLa' dataset available on the [cell tracking challenge](http://celltrackingchallenge.net) website. As in the main paper from 2015, a pixel-wise loss weight-map was calculated to force the network to learn to separate cells at cell borders. The generated segmentation mask with separation borders (c) and the weight loss map (d) can be seen below together with an input image (a) and its corresponding ground truth (b). 

![figure_1](https://gits-15.sys.kth.se/storage/user/6883/files/52f90580-7910-11e9-8a71-f97d29793d84)

## Directory Structure
The code is applicable to all seven 2D+time datasets in the cell segmentation challenge. The code can only be run if at least one of the datasets from the challenge is downloaded and structured in accordance to our file system. The image-files and their supplied ground truths are quite big, and aren't uploaded on this github page. Note that we're using 'DIC-C2DH-HeLa' as our working data set. The images in the datasets comes from two different film sequences and we copied the first segmentation masks and the corresponding images from sequence 01 (image 't002f.tif') and 02 (image 't006f.tif') and put them into a validation folder with the same internal structure as the training and testing folders, but here we only used a 01 sequence and renamed the copied images to 't000f.tif' and 't001.tif', respectively. This 'seed' validation dataset was then used to create bigger validation set using impemented transformations in the file 'validation_generator.py'. The structure of the dataset folders are given on the cell segmentation challenge website. For reference, see the file tree diagram below:

<p align="center">
  <img width="446" alt="skarmavbild 2019-05-18 kl 14 16 07" 
       src="https://gits-15.sys.kth.se/storage/user/6883/files/8753dc80-7977-11e9-988f-fe7c2f52570a">
</p>




