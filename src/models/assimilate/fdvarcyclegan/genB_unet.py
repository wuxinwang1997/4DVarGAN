# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from src.utils.model_utils import PeriodicPad2d

class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self,x):
        loss_ = torch.nansum(x**2 , dim=-1)
        loss_ = torch.nansum(loss_, dim=-1)
        loss_ = torch.nansum(loss_, dim=1)

        return loss_

class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dropout: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers =  [PeriodicPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dropout:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers += [PeriodicPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return x + self.net(x)

class UNet(nn.Module):
    """Implements the ViT model,

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        out_channels: int = 1,
        apply_dropout: bool = True,
        num_downsampling: int = 2,
        num_resnet_blocks: int = 4,
        init_type: str = 'normal',
        init_gain: float = 0.02,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        f = 1
        num_downsampling = num_downsampling
        num_resnet_blocks = num_resnet_blocks
        self.init_type = init_type
        self.init_gain = init_gain
        
        conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=7, stride=1)
        self.layers = [PeriodicPad2d(3), conv, nn.InstanceNorm2d(hidden_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(hidden_channels * f, hidden_channels * 2 * f, kernel_size=3, stride=2)
            self.layers += [PeriodicPad2d(1), conv, nn.InstanceNorm2d(hidden_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock(in_channels = hidden_channels * f, apply_dropout = apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(hidden_channels * f, hidden_channels * (f//2), 3, 2, padding=1, output_padding=1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1)
        self.layers += [PeriodicPad2d(3), conv]

        self.net = nn.Sequential(*self.layers)

    def init_module(self, m):
        cls_name = m.__class__.__name__
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        preds = self.net(x)

        return preds
