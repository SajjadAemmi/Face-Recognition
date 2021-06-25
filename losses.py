import torch
from torch import nn


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        cosine.acos_()
        cosine += self.m
        cosine.cos_().mul_(self.s)
        return cosine
