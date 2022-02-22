import torch
from torch import nn
from torchvision.models import alexnet
from models.sr_base_model import SRCNN
from models.base_model import BaseModel

class srnet(BaseModel):
    def __init__(self, num_channels=3, base_filter=64, upscale_factor=4):
        super().__init__()
        base_srcnn = SRCNN(num_channels=num_channels, base_filter=base_filter, upscale_factor=upscale_factor)
        self.shared_cnn_layers = base_srcnn.features

        self.old_layers = base_srcnn.classifier
        self.new_layers = nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, _input):
        features_out = self.shared_cnn_layers(_input)

        # Old task branch
        old_task_outputs = self.old_layers(features_out)
        # New task branch
        new_task_outputs = self.new_layers(features_out)
        outputs = torch.cat((old_task_outputs, new_task_outputs), dim=1)
        return outputs, old_task_outputs
