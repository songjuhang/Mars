import torch.nn as nn
from model.base.components import Conv, C2f

class Backbone(nn.Module):
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # 定义通道数
        c1 = int(128 * w)  # 128
        c2 = int(256 * w)  # 256
        c3 = int(512 * w)  # 512
        c4 = int(512 * w * r)  # 1024/768

        # Stem
        self.stem = Conv(3, c1, 3, 2)  # (B, c1, 320, 320)

        # Stage 1 (P2/4) 
        self.stage1_conv = Conv(c1, c2, 3, 2)
        self.stage1_c2f = C2f(c2, c2, n)

        # Stage 2 (P3/8)
        self.stage2_conv = Conv(c2, c3, 3, 2) 
        self.stage2_c2f = C2f(c3, c3, n)

        # Stage 3 (P4/16)
        self.stage3_conv = Conv(c3, c3, 3, 2)
        self.stage3_c2f = C2f(c3, c3, n)

        # Stage 4 (P5/32)
        self.stage4_conv = Conv(c3, c4, 3, 2)
        self.stage4_c2f = C2f(c4, c4, n)

    def forward(self, x):
        # Stem
        x = self.stem(x)  # (B, c1, 320, 320)

        # Stage 1 (P2/4)
        x = self.stage1_conv(x)  # (B, c2, 160, 160) 
        feat0 = self.stage1_c2f(x)  # (B, c2, 160, 160)

        # Stage 2 (P3/8)
        x = self.stage2_conv(feat0)  # (B, c3, 80, 80)
        feat1 = self.stage2_c2f(x)  # (B, c3, 80, 80)

        # Stage 3 (P4/16)
        x = self.stage3_conv(feat1)  # (B, c3, 40, 40)
        feat2 = self.stage3_c2f(x)  # (B, c3, 40, 40)

        # Stage 4 (P5/32)
        x = self.stage4_conv(feat2)  # (B, c4, 20, 20)
        feat3 = self.stage4_c2f(x)  # (B, c4, 20, 20)

        return feat0, feat1, feat2, feat3