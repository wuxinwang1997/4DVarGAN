#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wuxinwang
"""

from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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

    def forward(self, xb, obs, grad):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        preds = self.net(torch.concat([xb, obs, grad], dim=1))

        return preds

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array(shape_data)

    def forward(self, x, y, mask):
        dyout = (x - y) * mask

        return dyout

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarGAN(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, shape_data, adaptive):
        super(Solver_Grad_4DVarGAN, self).__init__()
        self.phi_r         = phi_r
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.loss = L2Norm()
        self.preds = []
        self.adaptive = adaptive

    def forward(self, x, yobs, mask, *internal_state):
        return self.solve(x, yobs, mask, *internal_state)

    def solve(self, x_0, obs, mask):
        x_k = torch.mul(x_0,1.)
        x_k_plus_1 = None
        self.preds = []
        x_k_plus_1 = self.solver_step(x_k, obs, mask)

        return x_k_plus_1

    def solver_step(self, x_k, obs, mask):
        xf, var_cost_grad, scale = self.var_cost(x_k, obs, mask)
        normgrad = torch.sqrt(torch.mean(var_cost_grad**2, dim=(1,2,3), keepdim=True))
        obs = mask * obs #+ (1-mask) * xf
        x_inc = self.model_Grad(x_k, obs, var_cost_grad / normgrad)
        x_k_plus_1 = x_k + x_inc * scale
        return x_k_plus_1

    def var_cost(self, xb, yobs, mask):
        self.preds.append(xb)
        for i in np.arange(1, yobs.shape[1]):
            self.preds.append(self.phi_r(self.preds[i-1]))
        preds = torch.concat(self.preds, dim=1)
        dy = self.model_H(preds, yobs, mask)
        loss = self.loss(dy)
        var_cost_grad = torch.autograd.grad(loss, xb, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        if self.adaptive:
            scale = loss / torch.count_nonzero(torch.flatten(mask[:,:,:,:], start_dim=1), dim=1)
            return preds, var_cost_grad, torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scale, dim=-1), dim=-1), dim=-1)
        else:
            return preds, var_cost_grad, 1 