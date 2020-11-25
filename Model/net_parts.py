import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, input_channels, output_channels, batch_norm=False):
        super(Layer, self).__init__()
        self.parts = nn.ModuleList([nn.Conv2d(input_channels, output_channels, 3, padding=0)])

        if batch_norm:
            self.parts.append(nn.BatchNorm2d(output_channels))

        self.parts.extend([nn.ReLU(), nn.Conv2d(output_channels, output_channels, 3, padding=0)])

        if batch_norm:
            self.parts.append(nn.BatchNorm2d(output_channels))

        self.parts.append(nn.ReLU())


    def forward(self, x):
        for op in self.parts:
            x = op(x)
        return x


class EncodingLayer(nn.Module):
    def __init__(self, input_channels, output_channels, batch_norm=False):
        super(EncodingLayer, self).__init__()
        self.conv_steps = Layer(input_channels, output_channels, batch_norm)

    # With relu
    def forward(self, x):
        x1 = F.max_pool2d(x, 2)
        x2 = self.conv_steps(x1)
        return x2


class DecodingLayer(nn.Module):
    def __init__(self, input_channels, output_channels, batch_norm=False, upsampling_type=True):
        super(DecodingLayer, self).__init__()

        if upsampling_type:
            self.decoding_layer = nn.ConvTranspose2d(input_channels,output_channels, 2, stride=2)
        else:
            self.decoding_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(input_channels, output_channels, kernel_size=1, )
            )

        self.conv_layer = Layer(input_channels, output_channels, batch_norm)


    def forward(self, x1, x2):
        x1 = self.decoding_layer(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Crop image
        x2 = x2[: , :, diffY // 2 : -diffY // 2, diffX // 2 : -diffX // 2]

        x = torch.cat([x2, x1], dim=1) # Concatenate image
        x = self.conv_layer(x) # Convolute the images
        return x


class FinalLayer(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(FinalLayer, self).__init__()
        self.final_conv = nn.Conv2d(input_channels, n_classes, 1)

    def forward(self, x):
        x = self.final_conv(x)
        return x
