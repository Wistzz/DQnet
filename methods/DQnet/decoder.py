import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .base import ConvBNReLU



class Simple_fuse(nn.Module):
    def __init__(self, feature_channels, out):
        super().__init__()

        self.c0_down = nn.Conv2d(feature_channels[0], out, kernel_size=1, stride=1, padding=0)
        self.c1_down = nn.Conv2d(feature_channels[1], out, kernel_size=1, stride=1, padding=0)
        self.c2_down = nn.Conv2d(feature_channels[2], out, kernel_size=1, stride=1, padding=0)
        self.c3_down = nn.Conv2d(feature_channels[3], out, kernel_size=1, stride=1, padding=0)
        self.c4_down = nn.Conv2d(feature_channels[4], out, kernel_size=1, stride=1, padding=0)


    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c0, c1, c2, c3, c4 = xs
        c0 = self.c0_down(c0)
        c1 = self.c1_down(c1)
        c2 = self.c2_down(c2)
        c3 = self.c3_down(c3)
        c4 = self.c4_down(c4)
        # c5 = self.c5_down(c5)        

        return [c0, c1, c2, c3, c4]


class SimpleHead(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, out=256):
        super(SimpleHead, self).__init__()

        feature_channels = [256,512,1024,2048,768]#[768,768,768,768]#
        self.fuse = Simple_fuse(feature_channels, out)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(4*out, out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(out, num_classes, kernel_size=3, padding=1)

    def forward(self, features):
        features = self.fuse(features)


        P = []
        P.append(features[-1])
        P.extend([up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)-1))])     
        decode_size = P[-1].size()[2:]
        P[:-1] = [F.interpolate(feature, size=decode_size, mode='bicubic', align_corners=True) for feature in P[:-1]]
        x = self.conv_fusion(torch.cat((P), dim=1))
        img_size = [i*4 for i in decode_size]
        x = self.head(x)
        x = F.interpolate(x, size=img_size, mode='bicubic')   


        return x


class AuxHead(nn.Module):
    def __init__(self, ):
        super(AuxHead, self).__init__()
        self.head = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=3, padding=1),\
            nn.Conv2d(256, 1, kernel_size=3, padding=1),\
            nn.Conv2d(512, 1, kernel_size=3, padding=1),\
            nn.Conv2d(1024, 1, kernel_size=3, padding=1)])
    def forward(self, features):
        x = [self.head[i](features[i]).sigmoid() for i in range(len(features))]
        decode_size = x[0].size()[2:]
        img_size = [i*4 for i in decode_size]
        x = [F.interpolate(feature, size=img_size, mode='bicubic') for feature in x]
        return x


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bicubic', align_corners=True) + y

