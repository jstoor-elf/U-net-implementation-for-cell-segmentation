
import torch.nn as nn
import torch.nn.functional as F

from .net_parts import *

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, f_depth=64, batch_norm=True):
        super(Unet, self).__init__()
        self.layer_1  = Layer(n_channels, f_depth * 1, batch_norm)
        self.layer_2  = EncodingLayer(f_depth * 1, f_depth * 2, batch_norm)
        self.layer_3  = EncodingLayer(f_depth * 2, f_depth * 4, batch_norm)
        self.layer_4  = EncodingLayer(f_depth * 4, f_depth * 8, batch_norm)
        self.layer_5  = EncodingLayer(f_depth * 8, f_depth *16, batch_norm)
        self.layer_d  = nn.Dropout(0.5)
        self.layer_6  = DecodingLayer(f_depth *16, f_depth * 8, batch_norm)
        self.layer_7  = DecodingLayer(f_depth * 8, f_depth * 4, batch_norm)
        self.layer_8  = DecodingLayer(f_depth * 4, f_depth * 2, batch_norm)
        self.layer_9  = DecodingLayer(f_depth * 2, f_depth * 1, batch_norm)
        self.layer_10 = FinalLayer(f_depth * 1, n_classes)


    def forward(self, x):
        # Decoder
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)
        x4 = self.layer_4(x3)
        x = self.layer_5(x4)

        # Dropout
        x = self.layer_d(x)

        # Encoder
        x = self.layer_6(x, x4)
        x = self.layer_7(x, x3)
        x = self.layer_8(x, x2)
        x = self.layer_9(x, x1)

        # Final layer and softmax
        x = self.layer_10(x)
        return x
