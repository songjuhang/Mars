import torch
import torch.nn as nn
from model.base.components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2
        self.ch = (int(64 * w), int(128 * w), int(256 * w), int(512 * w), int(512 * w * r))

        # P1/2
        self.conv1 = Conv(3, self.ch[0], self.kernelSize, self.stride)
        
        # P2/4
        self.conv2 = Conv(self.ch[0], self.ch[1], self.kernelSize, self.stride)
        self.c2f2 = C2f(self.ch[1], self.ch[1], n)
        
        # P3/8
        self.conv3 = Conv(self.ch[1], self.ch[2], self.kernelSize, self.stride)
        self.c2f3 = C2f(self.ch[2], self.ch[2], n * 2)
        
        # P4/16
        self.conv4 = Conv(self.ch[2], self.ch[3], self.kernelSize, self.stride)
        self.c2f4 = C2f(self.ch[3], self.ch[3], n * 2)
        
        # P5/32
        self.conv5 = Conv(self.ch[3], self.ch[4], self.kernelSize, self.stride)
        self.c2f5 = C2f(self.ch[4], self.ch[4], n)
        self.sppf = SPPF(self.ch[4], self.ch[4])

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 256 * w, 80, 80)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40) 
            feat3: (B, 512 * w * r, 20, 20)
        """
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        # backbone
        feat0 = self.conv1(x)           # P1/2
        feat1 = self.c2f2(
            self.conv2(feat0)           # P2/4
        )
        feat2 = self.c2f3(
            self.conv3(feat1)           # P3/8
        )
        feat3 = self.c2f4(
            self.conv4(feat2)           # P4/16
        )
        feat4 = self.sppf(
            self.c2f5(
                self.conv5(feat3)       # P5/32
            )
        )
        return feat1, feat2, feat3, feat4#问题