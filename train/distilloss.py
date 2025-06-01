import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

class CWDLoss(nn.Module):
    """Channel-wise Distillation Loss (L2范数)"""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, sfeats, tfeats):
        loss = 0.0
        for s, t in zip(sfeats, tfeats):
            loss += F.mse_loss(s, t)
        return loss

class ResponseLoss(nn.Module):
    """Response-based Distillation Loss (KL散度)"""
    def __init__(self, device, nc, teacherClassIndexes):
        super().__init__()
        self.device = device
        self.nc = nc
        self.teacherClassIndexes = teacherClassIndexes

    def forward(self, sresponse, tresponse):
        # 只对分类部分做KL散度
        loss = 0.0
        for s, t in zip(sresponse, tresponse):
            # s, t: (B, regMax*4 + nc, H, W)
            s_cls = s[:, -self.nc:, :, :]
            t_cls = t[:, -self.nc:, :, :]
            s_log_prob = F.log_softmax(s_cls, dim=1)
            t_prob = F.softmax(t_cls, dim=1)
            loss += F.kl_div(s_log_prob, t_prob, reduction='batchmean')
        return loss

class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        from train.loss import DetectionLoss
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(self.mcfg.device)
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes)

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0]: 学生网络输出 (list of feature maps)
        rawPreds[1]: 教师网络输出 (list of feature maps)
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]
        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]
        return loss.sum()