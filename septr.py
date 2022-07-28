# This code is released under the CC BY-SA 4.0 license.

import torch
from torch import nn
from septr_block import SepTrBlock


class SeparableTr(nn.Module):
    def __init__(self, channels=1, input_size=(128, 128), num_classes=35, depth=3, heads=5, mlp_dim=256, dim_head=256,
                 down_sample_input=None, dim=256):
        super().__init__()
        inner_channels = channels

        self.transformer = nn.ModuleList()

        if depth < 1:
            raise Exception("Depth cannot be smaller than 1!")

        self.transformer.append(
            SepTrBlock(channels=inner_channels, input_size=input_size, heads=heads, mlp_dim=mlp_dim,
                       dim_head=dim_head, down_sample_input=down_sample_input, dim=dim, project=True)
        )

        for i in range(1, depth):
            self.transformer.append(
                SepTrBlock(channels=inner_channels, input_size=input_size, heads=heads, mlp_dim=mlp_dim,
                           dim_head=dim_head, down_sample_input=down_sample_input, dim=dim, project=False)
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x, cls_token = self.transformer[0](x, self.cls_token)

        for i in range(1, len(self.transformer)):
            x, cls_token = self.transformer[i](x, cls_token)

        cls_token = cls_token[:, 0, 0, :]
        x = self.fc(cls_token)
        return x
