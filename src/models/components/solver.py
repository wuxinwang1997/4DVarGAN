import torch
import torch.nn as nn
import numpy as np

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic
        #self.correlate_noise = CorrelateNoise(input_size, 10)
        #self.regularize_variance = RegularizeVariance(input_size, 10)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        """
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(device)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        """
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
        """
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_,z), prev_hidden), 1)
        """

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

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()

    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()

    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,periodicBnd=False,DimLSTM=0,rateDropout=0.,stochastic=False):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.dim_state  = 5*self.shape[0]
            else :
                self.dim_state  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer     = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)
        self.stochastic=stochastic

        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.dim_state,3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.dim_state, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.dim_state, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)
    def _make_LSTMGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],self.dim_state,3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic))

        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):
        # compute gradient
        grad  = grad / gradnorm
        grad  = self.dropout( grad )

        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,grad.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,grad.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,grad.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])

            hidden = hidden_[:,:,dB:grad.size(2)+dB,:]
            cell   = cell_[:,:,dB:grad.size(2)+dB,:]
        else:
            if hidden is None:
                hidden,cell = self.lstm(grad,None)
            else:
                hidden,cell = self.lstm(grad,[hidden,cell])

        grad = self.dropout( hidden )
        grad = self.convLayer( grad )

        return grad,hidden,cell

# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData,dim_obs=1,dim_obs_channel=0,dim_state=0):
        super(Model_Var_Cost, self).__init__()
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs        = dim_obs
        if dim_state > 0 :
            self.dim_state      = dim_state
        else:
            self.dim_state      = ShapeData[0]

        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dim_obs_channel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0]))))
            self.dim_obs_channel  = ShapeData[0] * np.ones((self.dim_obs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))

        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi

    def forward(self, dx, dy):

        loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
        if self.dim_obs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.dim_obs):
                loss +=  (
                    self.alphaObs[kk]**2
                    * self.normObs(
                        dy[kk],
                        self.WObs[kk,0:dy[kk].size(1)]**2,
                        self.epsObs[kk]
                    )
                )

        return loss

class Model_H(torch.nn.Module): # 观测算子
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array(shape_data)

    def forward(self, x, y, mask):
        dyout = (x - y) * mask # 输出的是观测位置的dx

        return dyout

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    NORMS = {
            'l1': Model_WeightedL1Norm,
            'l2': Model_WeightedL2Norm,
    }
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, shape_data,n_iter_grad, stochastic=False):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r

        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        else:
            m_NormObs = self.NORMS[m_NormObs]()
        if m_NormPhi == None:
            m_NormPhi = Model_WeightedL2Norm()
        else:
            m_NormPhi = self.NORMS[m_NormPhi]()
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, shape_data, mod_H.dim_obs, mod_H.dim_obs_channel)

        self.stochastic = stochastic

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)

    def forward(self, x, yobs, mask, *internal_state):
        return self.solve(x, yobs, mask, *internal_state)

    def solve(self, x_0, obs, mask, hidden=None, cell=None, normgrad_=0.):
        x_k = torch.mul(x_0,1.)
        x_k_plus_1 = None
        for _ in range(self.n_grad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask,hidden, cell, normgrad_)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden, cell, normgrad_

    def solver_step(self, x_k, obs, mask, hidden, cell,normgrad = 0.):
        _, var_cost_grad= self.var_cost(x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self, x, yobs, mask):
        dy = self.model_H(x, yobs, mask)
        dx = x - self.phi_r(x)

        loss = self.model_VarCost(dx ,dy)

        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad


class Solver_4DVarGAN(nn.Module):
    NORMS = {
            'l1': Model_WeightedL1Norm,
            'l2': Model_WeightedL2Norm,
    }
    def __init__(self ,phi_r,mod_H, m_GAN, m_NormObs, m_NormPhi, shape_data):
        super(Solver_4DVarGAN, self).__init__()
        self.phi_r         = phi_r

        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        else:
            m_NormObs = self.NORMS[m_NormObs]()
        if m_NormPhi == None:
            m_NormPhi = Model_WeightedL2Norm()
        else:
            m_NormPhi = self.NORMS[m_NormPhi]()
        self.shape_data = shape_data
        self.model_H = mod_H
        self.m_GAN = m_GAN
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, shape_data, mod_H.dim_obs, mod_H.dim_obs_channel)

    def forward(self, x, yobs, mask):
        return self.solve(x, yobs, mask)

    def solve(self, xb, obs, mask):
        xa = self.solver_step(xb, obs, mask)
        return xa

    def solver_step(self, xb, obs, mask):
        b, c, h, w = xb.shape
        pred_x = self.phi_r(xb)
        dy = self.model_H(torch.concat([xb, pred_x],dim=1), obs, mask)
        _, var_cost_grad= self.var_cost(xb, pred_x, dy, mask)
        obs = mask * obs + (1-mask)*torch.concat([xb, pred_x], dim=1)
        x_inc = self.m_GAN(torch.concat([xb, pred_x, obs, var_cost_grad], dim=1))
        dy_ = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.sum(dy[:,0,:,:]**2, dim=[1,2]),dim=-1),dim=-1),dim=-1)
        return xb + x_inc * dy_/torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.count_nonzero(torch.flatten(mask[:,0,:,:], start_dim=1), dim=1),dim=-1),dim=-1),dim=-1)
    
    # # 以下方案是可行的，但是随着观测数目的减少，效果急剧变差
    # def solver_step(self, xb, obs, mask):
    #     b, c, h, w = xb.shape
    #     dy = mask[:,0:1,:,:]*(obs[:,0:1,:,:] - xb)
    #     x_inc = self.m_GAN(torch.concat([xb, obs], dim=1))
    #     dy_ = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.sum(dy[:,0,:,:]**2, dim=[1,2]),dim=-1),dim=-1),dim=-1)
    #     return xb + x_inc * dy_/torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.count_nonzero(torch.flatten(mask[:,0,:,:], start_dim=1), dim=1),dim=-1),dim=-1),dim=-1)

    def var_cost(self, xb, pred_x, dy, mask):
        loss = self.model_VarCost(xb-xb, dy)
        var_cost_grad = torch.autograd.grad(loss, xb, create_graph=True)[0]
        return loss, var_cost_grad