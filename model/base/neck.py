import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.components import Conv, C2f


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2
        # 保持原有的通道设置
        self.ch = (int(256 * w), int(512 * w), int(512 * w * r))

        # 修改通道数以匹配输入输出
        self.cv1 = Conv(self.ch[2], self.ch[1], 1, 1)  # 从大尺度到中尺度的上采样
        self.c2f1 = C2f(self.ch[1] * 2, self.ch[1], n) # 特征融合
        
        self.cv2 = Conv(self.ch[1], self.ch[0], 1, 1)  # 从中尺度到小尺度的上采样
        self.c2f2 = C2f(self.ch[0] * 2, self.ch[0], n) # 特征融合
        
        # 下采样路径的通道数需要匹配
        self.cv3 = Conv(self.ch[0], self.ch[1], self.kernelSize, self.stride) # 改为输出ch[1]通道
        self.c2f3 = C2f(self.ch[1] * 2, self.ch[1], n) # 保持一致的输出通道数
        
        self.cv4 = Conv(self.ch[1], self.ch[2], self.kernelSize, self.stride) # 改为输出ch[2]通道
        self.c2f4 = C2f(self.ch[2] * 2, self.ch[2], n) # 最终输出通道数

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # upsample path (from bottom to top)
        up1 = self.cv1(feat3)                          # (B, 512*w, 20, 20)
        up1 = F.interpolate(up1, scale_factor=2)       # (B, 512*w, 40, 40)
        feat_y = self.c2f1(torch.cat([up1, feat2], 1)) # (B, 512*w, 40, 40)

        up2 = self.cv2(feat_y)                         # (B, 256*w, 40, 40)
        up2 = F.interpolate(up2, scale_factor=2)       # (B, 256*w, 80, 80)
        feat_x = self.c2f2(torch.cat([up2, feat1], 1)) # (B, 256*w, 80, 80)

        # downsample path (from top to bottom)
        down1 = self.cv3(feat_x)                       # (B, 256*w, 40, 40)
        feat_c = self.c2f3(torch.cat([down1, feat_y], 1)) # (B, 512*w, 40, 40)

        down2 = self.cv4(feat_c)                       # (B, 512*w, 20, 20)
        feat_z = self.c2f4(torch.cat([down2, feat3], 1)) # (B, 512*w*r, 20, 20)

        return feat_c, feat_x, feat_y, feat_z