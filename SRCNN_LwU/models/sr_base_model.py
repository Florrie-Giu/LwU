import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F



class SRCNN(torch.nn.Module):
    def __init__(self, num_channels, base_filter,  upscale_factor=4):
        super(SRCNN, self).__init__()

        self.features = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.classifier = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )


    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)    # 正态分布
        m.bias.data.zero_()   # 为零
