#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Pytorch ConvLSTM2D
class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=self.kernel_size, stride=1,
                                     padding=self.padding)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(input_.device),
                torch.autograd.Variable(torch.zeros(state_size)).to(input_.device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

# Gradient computation: subgradient assuming a prior of the form ||x-f(x)||^2 vs. autograd
class Compute_Grad(torch.nn.Module):
    def __init__(self, ShapeData, GradType):
        super(Compute_Grad, self).__init__()

        with torch.no_grad():
            self.GradType = GradType
            self.shape = ShapeData

        self.alphaObs = torch.nn.Parameter(torch.Tensor([1.]))
        self.alphaAE = torch.nn.Parameter(torch.Tensor([1.]))
        if (self.GradType == 3):
            self.NbEngTerms = 3
            self.alphaEngTerms = torch.nn.Parameter(torch.Tensor(np.ones((self.NbEngTerms, 1))))
            self.alphaEngTerms.requires_grad = True

        if (self.GradType == 2):
            self.alphaL1 = torch.nn.Parameter(torch.Tensor([1.]))
            self.alphaL2 = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x, xpred, xobs, mask):
        # compute gradient
        if self.GradType == 0:  ## subgradient for prior ||x-g(x)||^2
            grad = torch.add(xpred, -1., x)
            grad2 = torch.add(x, -1., xobs)
            grad = torch.add(grad, 1., grad2)
            grad = self.alphaAE * grad + self.alphaObs * grad2

        elif self.GradType == 1:  ## true gradient using autograd for prior ||x-g(x)||^2
            loss1 = torch.mean((xpred - x) ** 2)
            loss2 = torch.sum((xobs - torch.concat([x, xpred], dim=1)) ** 2 * mask) / torch.sum(mask)
            loss = self.alphaAE ** 2 * loss1 + self.alphaObs ** 2 * loss2
            # =loss  = loss / ( self.alphaAE**2 + self.alphaObs**2)

            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            # grad = torch.autograd.grad(loss,x)[0]

        elif self.GradType == 2:  ## true gradient using autograd for prior ||x-g(x)||
            loss1 = self.alphaL2 ** 2 * torch.mean((xpred - x) ** 2) + self.alphaL1 ** 2 * torch.mean(
                torch.abs(xpred - x))
            loss2 = torch.sum((xobs - torch.concat([x, xpred], dim=1)) ** 2 * mask) / torch.sum(mask)
            loss = self.alphaAE ** 2 * loss1 + self.alphaObs ** 2 * loss2
            # =loss  = loss / ( self.alphaAE**2 + self.alphaObs**2)

            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            # grad = torch.autograd.grad(loss,x)[0]

        elif self.GradType == 4:  ## true gradient using autograd for prior ||g(x)||^2
            loss1 = torch.mean(xpred ** 2)
            loss2 = torch.sum((xobs - xpred) ** 2 * mask) / torch.sum(mask)
            loss = self.alphaAE ** 2 * loss1 + self.alphaObs ** 2 * loss2
            # =loss  = loss / ( self.alphaAE**2 + self.alphaObs**2)

            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            # grad = torch.autograd.grad(loss,x)[0]

        elif self.GradType == 3:  ## true gradient using autograd for prior ||x-g1(x)||^2 + ||x-g2(x)||^2
            if len(self.shape) == 2:
                for ii in range(0, xpred.size(1)):
                    if (ii == 0):
                        loss1 = self.alphaEngTerms[ii] ** 2 * torch.mean(
                            (xpred[:, ii, :, :].view(-1, self.shape[0], self.shape[1]) - x) ** 2)
                    else:
                        loss1 += self.alphaEngTerms[ii] ** 2 * torch.mean(
                            (xpred[:, ii, :, :].view(-1, self.shape[0], self.shape[1]) - x) ** 2)
            else:
                if (ii == 0):
                    loss1 = self.alphaEngTerms[ii] ** 2 * torch.mean((xpred[:, 0:self.shape[0], :, :] - x) ** 2)
                else:
                    loss1 += self.alphaEngTerms[ii] ** 2 * torch.mean(
                        (xpred[:, ii * self.shape[0]:(ii + 1) * self.shape[0], :, :] - x) ** 2)

            loss2 = torch.sum((xobs - x) ** 2 * mask) / torch.sum(mask)
            loss = self.alphaAE ** 2 * loss1 + self.alphaObs ** 2 * loss2
            # =loss  = loss / ( self.alphaAE**2 + self.alphaObs**2)

            grad = torch.autograd.grad(loss, x, create_graph=True)[0]

        # Check is this is needed or not
        grad.retain_grad()

        return grad

# Gradient-based minimization using a fixed-step descent
class model_GradUpdate0(torch.nn.Module):
    def __init__(self, ShapeData, GradType):
        super(model_GradUpdate0, self).__init__()

        with torch.no_grad():
            self.GradType = GradType
            self.shape = ShapeData
            self.delta = torch.nn.Parameter(torch.Tensor([1.]))
        self.compute_Grad = Compute_Grad(ShapeData, GradType)
        self.gradNet = self._make_ConvGrad()
        self.bn1 = torch.nn.BatchNorm2d(self.shape[0])

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2:  ## 1D Data
            layers.append(torch.nn.Conv1d(self.shape[0], self.shape[0], 1, padding=0, bias=False))
        elif len(self.shape) == 3:  ## 2D Data
            conv1 = torch.nn.Conv2d(self.shape[0], self.shape[0], (1, 1), padding=0, bias=False)
            # predefined parameters
            K = torch.Tensor([0.1]).view(1, 1, 1, 1)  # should be 0.1 is no bn is used
            conv1.weight = torch.nn.Parameter(K)
            layers.append(conv1)

        return torch.nn.Sequential(*layers)

    def forward(self, x, xpred, xobs, mask, gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred, xobs, mask)

        # update
        grad = self.gradNet(grad)

        return grad

# Gradient-based minimization using a CNN using a (sub)gradient as inputs
class model_GradUpdate1(torch.nn.Module):
    def __init__(self, ShapeData, GradType, periodicBnd=False):
        super(model_GradUpdate1, self).__init__()

        with torch.no_grad():
            self.GradType = GradType
            self.shape = ShapeData
            # self.delta     = torch.nn.Parameter(torch.Tensor([1e4]))
            self.PeriodicBnd = periodicBnd

            if ((self.PeriodicBnd == True) & (len(self.shape) == 2)):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
        self.compute_Grad = Compute_Grad(ShapeData, GradType)
        self.gradNet1 = self._make_ConvGrad()

        if len(self.shape) == 2:  ## 1D Data
            self.gradNet2 = torch.nn.Conv1d(self.shape[0], self.shape[0], 1, padding=0, bias=False)
            self.gradNet3 = torch.nn.Conv1d(self.shape[0], self.shape[0], 1, padding=0, bias=False)
            print(self.gradNet3.weight.size())
            K = torch.Tensor(np.identity(self.shape[0])).view(self.shape[0], self.shape[0], 1)
            self.gradNet3.weight = torch.nn.Parameter(K)
        elif len(self.shape) == 3:  ## 2D Data
            self.gradNet2 = torch.nn.Conv2d(self.shape[0], self.shape[0], (1, 1), padding=0, bias=False)
            self.gradNet3 = torch.nn.Conv2d(self.shape[0], self.shape[0], (1, 1), padding=0, bias=False)
            # self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
            # self.bn2           = torch.nn.BatchNorm2d(self.shape[0])
            K = torch.Tensor(np.identity(self.shape[0])).view(self.shape[0], self.shape[0], 1, 1)
            self.gradNet3.weight = torch.nn.Parameter(K)

        # with torch.enable_grad():
        # with torch.enable_grad():

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2:  ## 1D Data
            layers.append(torch.nn.Conv1d(2 * self.shape[0], 8 * self.shape[0], 3, padding=1, bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv1d(8 * self.shape[0], 16 * self.shape[0], 3, padding=1, bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv1d(16 * self.shape[0], self.shape[0], 3, padding=1, bias=False))
        elif len(self.shape) == 3:  ## 2D Data
            # layers.append(torch.nn.Conv2d(2*self.shape[0], self.shape[0], (3,3), padding=1,bias=False))
            layers.append(torch.nn.Conv2d(2 * self.shape[0], 8 * self.shape[0], (3, 3), padding=1, bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(8 * self.shape[0], 16 * self.shape[0], (3, 3), padding=1, bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(16 * self.shape[0], self.shape[0], (3, 3), padding=1, bias=False))

        return torch.nn.Sequential(*layers)

    def forward(self, x, xpred, xobs, mask, grad_old, gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred, xobs, mask)

        # grad = grad /self.ScaleGrad
        # grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        grad = grad / gradnorm

        if grad_old is None:
            grad_old = torch.randn(grad.size()).to(x.device)  ## Here device is global variable to be checked

        # boundary conditons
        if self.PeriodicBnd == True:
            dB = 7
            #
            grad_ = torch.cat((grad[:, :, x.size(2) - dB:, :], grad, grad[:, :, 0:dB, :]), dim=2)

            grad_old_ = torch.cat((grad_old[:, :, x.size(2) - dB:, :], grad_old, grad_old[:, :, 0:dB, :]), dim=2)

            gradAll = torch.cat((grad_old_, grad_), 1)

            dgrad = self.gradNet1(gradAll)
            grad = grad + self.gradNet2(dgrad[:, :, dB:x.size(2) + dB, :])

        else:
            gradAll = torch.cat((grad_old, grad), 1)

            dgrad = self.gradNet1(gradAll)
            grad = grad + self.gradNet2(dgrad)

        grad = 5. * torch.atan(0.2 * self.gradNet3(grad))

        return grad


# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdate2(torch.nn.Module):
    def __init__(self, ShapeData, GradType, periodicBnd=False):
        super(model_GradUpdate2, self).__init__()

        with torch.no_grad():
            self.GradType = GradType
            self.shape = ShapeData
            self.DimState = 5 * self.shape[0]
            self.PeriodicBnd = periodicBnd
            if ((self.PeriodicBnd == True) & (len(self.shape) == 2)):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
        self.compute_Grad = Compute_Grad(ShapeData, GradType)
        self.convLayer = self._make_ConvGrad()
        # self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
        # self.lstm            = self._make_LSTMGrad()
        K = torch.Tensor([0.1]).view(1, 1, 1, 1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.lstm = ConvLSTM2d(self.shape[0], self.DimState, 3)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2:  ## 1D Data
            layers.append(torch.nn.Conv1d(5 * self.shape[0], self.shape[0], 1, padding=0, bias=False))
        elif len(self.shape) == 3:  ## 2D Data
            layers.append(torch.nn.Conv2d(5 * self.shape[0], self.shape[0], (1, 1), padding=0, bias=False))

        return torch.nn.Sequential(*layers)

    def _make_LSTMGrad(self):
        layers = []

        layers.append(ConvLSTM2d(self.shape[0], 5 * self.shape[0], 3))

        return torch.nn.Sequential(*layers)

    def forward(self, x, xpred, xobs, mask, hidden, cell, gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred, xobs, mask)

        # grad = grad /self.ScaleGrad
        # grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        # grad = self.bn1(grad)
        grad = grad / gradnorm

        if self.PeriodicBnd == True:
            dB = 7
            #
            grad_ = torch.cat((grad[:, :, x.size(2) - dB:, :], grad, grad[:, :, 0:dB, :]), dim=2)
            if hidden is None:
                hidden_, cell_ = self.lstm(grad_, None)
            else:
                hidden_ = torch.cat((hidden[:, :, x.size(2) - dB:, :], hidden, hidden[:, :, 0:dB, :]), dim=2)
                cell_ = torch.cat((cell[:, :, x.size(2) - dB:, :], cell, cell[:, :, 0:dB, :]), dim=2)
                hidden_, cell_ = self.lstm(grad_, [hidden_, cell_])

            hidden = hidden_[:, :, dB:x.size(2) + dB, :]
            cell = cell_[:, :, dB:x.size(2) + dB, :]
        else:
            if hidden is None:
                hidden, cell = self.lstm(grad, None)
            else:
                hidden, cell = self.lstm(grad, [hidden, cell])

        grad = self.convLayer(hidden)

        return grad, hidden, cell


class model_GradUpdate3(torch.nn.Module):
    def __init__(self, ShapeData, GradType, dimState=30):
        super(model_GradUpdate3, self).__init__()

        with torch.no_grad():
            self.GradType = GradType
            self.shape = ShapeData
            self.DimState = dimState
            self.layer_dim = 1
        self.compute_Grad = Compute_Grad(ShapeData, GradType)

        if len(self.shape) == 2:  ## 1D Data
            self.convLayer = torch.Linear(dimState, self.shape[0] * self.shape[1])
            self.lstm = torch.nn.LSTM(self.shape[0] * self.shape[1], self.DimState, self.layer_dim)
        else:
            self.convLayer = torch.Linear(dimState, self.shape[0] * self.shape[1] * self.shape[2])
            self.lstm = torch.nn.LSTM(self.shape[0] * self.shape[1] * self.shape[2], self.DimState, self.layer_dim)

    def forward(self, x, xpred, xobs, mask, hidden, cell, gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred, xobs, mask)

        # grad = grad /self.ScaleGrad
        # grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        # grad = self.bn1(grad)
        grad = grad / gradnorm

        if len(self.shape) == 2:  ## 1D Data
            grad = grad.view(-1, 1, self.shape[0] * self.shape[1])
        else:
            grad = grad.view(-1, 1, self.shape[0] * self.shape[1] * self.shape[2])

        if hidden is None:
            output, (hidden, cell) = self.lstm(grad, None)
        else:
            output, (hidden, cell) = self.lstm(grad, (hidden, cell))

        grad = self.convLayer(output)
        if len(self.shape) == 2:  ## 1D Data
            grad = grad.view(-1, self.shape[0], self.shape[1])
        else:
            grad = grad.view(-1, self.shape[0], self.shape[1], self.shape[2])

        return grad, hidden, cell

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using a number of
# fixed-point iterations prior to a gradient-based minimization
class Model_4DVarNN_GradFP(torch.nn.Module):
    def __init__(self, mod_AE, ShapeData, NiterProjection, NiterGrad, GradType, OptimType, InterpFlag=False,
                 periodicBnd=False):
        # def __init__(self,mod_AE,GradType,OptimType):
        super(Model_4DVarNN_GradFP, self).__init__()

        self.model_AE = mod_AE

        with torch.no_grad():
            print('Opitm type %d' % OptimType)
            self.OptimType = OptimType
            self.NProjFP = int(NiterProjection)
            self.NGrad = int(NiterGrad)
            self.InterpFlag = InterpFlag
            self.periodicBnd = periodicBnd

        if OptimType == 0:
            self.model_Grad = model_GradUpdate0(ShapeData, GradType)
        elif OptimType == 1:
            self.model_Grad = model_GradUpdate1(ShapeData, GradType, self.periodicBnd)
        elif OptimType == 2:
            self.model_Grad = model_GradUpdate2(ShapeData, GradType, self.periodicBnd)
        elif OptimType == 3:
            self.model_Grad = model_GradUpdate3(ShapeData, GradType, 30)

    def forward(self, x_inp, xobs, mask, g1=None, g2=None, normgrad=0.0):
        mask_ = torch.add(1.0, torch.mul(mask, -1.0))  # 1. - mask

        x = torch.mul(x_inp, 1.0)

        # fixed-point iterations
        if self.NProjFP > 0:
            for kk in range(0, self.NProjFP):
                # if NiterProjection > 0:
                #  x      = torch.mul(x_inp,1.0)
                #  for kk in range(0,NiterProjection):
                x_proj = self.model_AE(x)
                x_proj = torch.mul(x_proj, mask_)
                x = torch.mul(x, mask)
                x = torch.add(x, x_proj)

        # gradient iteration
        if self.NGrad > 0:
            # gradient normalisation
            # xpred = torch.zeros_like(xobs)
            # xpred[:, 0, :, :] = x[:, 0, :, :]
            # for i in range(xobs.shape[1] - 1):
            #     xpred[:, i+1, :, :] = self.model_AE(xpred[:, i:i+1, :, :])[:, 0, :, :]
            grad = self.model_Grad.compute_Grad(x, self.model_AE(x), xobs, mask)
            if normgrad == 0.:
                _normgrad = torch.sqrt(torch.mean(grad ** 2))
            else:
                _normgrad = normgrad
            for kk in range(0, self.NGrad):
                # AE pediction
                # xpred[:, 0, :, :] = x[:, 0, :, :]
                # for i in range(xobs.shape[1] - 1):
                #     xpred[:, i+1, :, :] = self.model_AE(xpred[:, i:i+1, :, :])[:, 0, :, :]
                xpred = self.model_AE(x)

                # gradient update
                if self.OptimType == 0:
                    grad = self.model_Grad(x, xpred, xobs, mask, _normgrad)

                elif self.OptimType == 1:
                    if kk == 0:
                        grad = self.model_Grad(x, xpred, xobs, mask, g1, _normgrad)
                    else:
                        grad = self.model_Grad(x, xpred, xobs, mask, grad_old, _normgrad)
                    grad_old = torch.mul(1., grad)

                elif self.OptimType == 2:
                    if kk == 0:
                        grad, hidden, cell = self.model_Grad(x, xpred, xobs, mask, g1, g2, _normgrad)
                    else:
                        grad, hidden, cell = self.model_Grad(x, xpred, xobs, mask, hidden, cell, _normgrad)

                # interpolation constraint
                if (self.InterpFlag == True):
                    # optimization update
                    xnew = x - grad
                    x = x * mask + xnew * mask_
                else:
                    # optimization update
                    x = x - grad

            if self.OptimType == 1:
                return x, grad_old, _normgrad
            if self.OptimType == 2:
                return x, hidden, cell, _normgrad
            else:
                return x, _normgrad
        else:
            _normgrad = 1.
            if self.OptimType == 1:
                return x, None, _normgrad
            if self.OptimType == 2:
                return x, None, None, _normgrad
            else:
                return x, _normgrad

def compute_WeightedLoss(x2, w):
    x2_msk = x2[:, w == 1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    loss2 = loss2 * w.sum()
    return loss2

    # loss_ = torch.nansum(x2**2 , dim = 3)
    ##loss_ = torch.nansum( loss_ , dim = 2)
    # loss_ = torch.nansum( loss_ , dim = 0)
    # loss_ = torch.nansum( loss_ * w )
    # loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1] )

    # return loss_


# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()

    def forward(self, x, w, eps=0.):
        loss_ = torch.nansum(x ** 2, dim=3)
        loss_ = torch.nansum(loss_, dim=2)
        loss_ = torch.nansum(loss_, dim=0)
        loss_ = torch.nansum(loss_ * w)
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1])

        return loss_


class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()

    def forward(self, x, w, eps):
        loss_ = torch.nansum(torch.sqrt(eps ** 2 + x ** 2), dim=3)
        loss_ = torch.nansum(loss_, dim=2)
        loss_ = torch.nansum(loss_, dim=0)
        loss_ = torch.nansum(loss_ * w)
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1])

        return loss_


class Model_WeightedLorenzNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedLorenzNorm, self).__init__()

    def forward(self, x, w, eps):
        loss_ = torch.nansum(torch.log(1. + eps ** 2 * x ** 2), dim=3)
        loss_ = torch.nansum(loss_, dim=2)
        loss_ = torch.nansum(loss_, dim=0)
        loss_ = torch.nansum(loss_ * w)
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1])

        return loss_


class Model_WeightedGMcLNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()

    def forward(self, x, w, eps):
        loss_ = torch.nansum(1.0 - torch.exp(- eps ** 2 * x ** 2), dim=3)
        loss_ = torch.nansum(loss_, dim=2)
        loss_ = torch.nansum(loss_, dim=0)
        loss_ = torch.nansum(loss_ * w)
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1])

        return loss_


def compute_WeightedL2Norm1D(x2, w):
    loss_ = torch.nansum(x2 ** 2, dim=2)
    loss_ = torch.nansum(loss_, dim=0)
    loss_ = torch.nansum(loss_ * w)
    loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1])

    return loss_


# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self, ShapeData, periodicBnd=False, DimLSTM=0, rateDropout=0.):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape = ShapeData
            if DimLSTM == 0:
                self.DimState = 5 * self.shape[0]
            else:
                self.DimState = DimLSTM
            self.PeriodicBnd = periodicBnd
            if ((self.PeriodicBnd == True) & (len(self.shape) == 2)):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1, 1, 1, 1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)

        self.lstm = ConvLSTM2d(self.shape[0], self.DimState, 3)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2:  ## 1D Data
            layers.append(torch.nn.Conv1d(self.DimState, self.shape[0], 1, padding=0, bias=False))
        elif len(self.shape) == 3:  ## 2D Data
            layers.append(torch.nn.Conv2d(self.DimState, self.shape[0], (1, 1), padding=0, bias=False))

        return torch.nn.Sequential(*layers)

    def _make_LSTMGrad(self):
        layers = []

        layers.append(ConvLSTM2d(self.shape[0], self.DimState, 3))

        return torch.nn.Sequential(*layers)

    def forward(self, hidden, cell, grad, gradnorm=1.0):

        # compute gradient
        grad = grad / gradnorm
        grad = self.dropout(grad)

        if self.PeriodicBnd == True:
            dB = 7
            #
            grad_ = torch.cat((grad[:, :, x.size(2) - dB:, :], grad, grad[:, :, 0:dB, :]), dim=2)
            if hidden is None:
                hidden_, cell_ = self.lstm(grad_, None)
            else:
                hidden_ = torch.cat((hidden[:, :, x.size(2) - dB:, :], hidden, hidden[:, :, 0:dB, :]), dim=2)
                cell_ = torch.cat((cell[:, :, x.size(2) - dB:, :], cell, cell[:, :, 0:dB, :]), dim=2)
                hidden_, cell_ = self.lstm(grad_, [hidden_, cell_])

            hidden_ = hidden_[:, :, dB:x.size(2) + dB, :]
            cell_ = cell_[:, :, dB:x.size(2) + dB, :]
        else:
            if hidden is None:
                hidden_, cell_ = self.lstm(grad, None)
            else:
                hidden_, cell_ = self.lstm(grad, [hidden, cell])

        grad = self.dropout(hidden_)
        grad = self.convLayer(grad)

        return grad, hidden_, cell_


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self, m_NormObs, m_NormPhi, ShapeData, DimObs=1, dimObsChannel=0, dimState=0):
        super(Model_Var_Cost, self).__init__()
        self.dimObsChannel = dimObsChannel
        self.DimObs = DimObs
        if dimState > 0:
            self.DimState = dimState
        else:
            self.DimState = ShapeData[0]

        # parameters for variational cost
        self.alphaObs = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.DimObs, 1))))
        self.alphaReg = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dimObsChannel[0] == 0:
            self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs, ShapeData[0]))))
            self.dimObsChannel = ShapeData[0] * np.ones((self.DimObs,))
        else:
            self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs, np.max(self.dimObsChannel)))))
        self.WReg = torch.nn.Parameter(torch.Tensor(np.ones(self.DimState, )))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.DimObs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))

        self.normObs = m_NormObs
        self.normPrior = m_NormPhi

    def forward(self, dx, dy):

        loss = self.alphaReg ** 2 * self.normPrior(dx, self.WReg ** 2, self.epsReg)

        if self.DimObs == 1:
            loss += self.alphaObs[0] ** 2 * self.normObs(dy, self.WObs[0, :] ** 2, self.epsObs[0])
        else:
            for kk in range(0, self.DimObs):
                loss += self.alphaObs[kk] ** 2 * self.normObs(dy[kk], self.WObs[kk, 0:dy[kk].size(1)] ** 2,
                                                              self.epsObs[kk])

        return loss