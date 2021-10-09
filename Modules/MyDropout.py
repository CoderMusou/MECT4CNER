import torch
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x
