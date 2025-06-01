import torch
import torch.nn as nn
from model.base.components import Conv, C2f, SPPF


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    YOLOv8 Neck with FPN + PAN structure
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # Channel configurations to match backbone output
        c2 = int(256 * w)  # feat1 channels  
        c3 = int(512 * w)  # feat2 channels  
        c4 = int(512 * w * r)  # feat3 channels

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # SPPF
        self.sppf = SPPF(c4, c4)
        
        # Top-down pathway
        self.reduce_layer2 = Conv(c4, c3, 1, 1)
        self.c2f2 = C2f(c3, c3, n)
        self.reduce_layer1 = Conv(c3, c2, 1, 1)
        self.c2f1 = C2f(c2, c2, n)

        # Bottom-up pathway
        self.downsample1 = Conv(c2, c3, 3, 2)
        self.downsample2 = Conv(c3, c4, 3, 2)

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            P4: (B, 512 * w, 40, 40)
            P3: (B, 256 * w, 80, 80)
            N4: (B, 512 * w, 40, 40)
            N5: (B, 512 * w * r, 20, 20)
        """
        # SPPF enhancement
        P5 = self.sppf(feat3)  # (B, c4, 20, 20)
        
        # Top-down pathway
        P5_up = self.upsample(self.reduce_layer2(P5))  # (B, c3, 40, 40)
        P4 = self.c2f2(P5_up)  # (B, c3, 40, 40)
        
        P4_up = self.upsample(self.reduce_layer1(P4))  # (B, c2, 80, 80)
        P3 = self.c2f1(P4_up)  # (B, c2, 80, 80)
        
        # Bottom-up pathway
        N4 = self.downsample1(P3)  # (B, c3, 40, 40)
        N4 = self.c2f2(N4)  # (B, c3, 40, 40)
        
        N5 = self.downsample2(N4)  # (B, c4, 20, 20)
        N5 = self.sppf(N5)  # (B, c4, 20, 20)

        return P4, P3, N4, N5