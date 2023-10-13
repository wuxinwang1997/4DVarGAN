import numpy as np
import torch
import copy

#### 4DVar implementation following Marcel Nonnenmacher's implementation ######
## Deep Emulators for Differentiation, Forecasting, and Parametrization in Earth Science Simulators

def Solve_Var4D(x_init, B_inv, R_inv, maxIter, model, dtmodel, obs, dt_obs, obs_masks, daw):
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache() 
    shape = x_init.shape
    x_init = x_init.detach()
    x_init = torch.reshape(x_init,[shape[-1]*shape[-2],1])
    x_analysis = copy.deepcopy(x_init)
    x_analysis.requires_grad = True
    vec_shape = x_analysis.shape
    obs_num = obs.shape[0]
    obs = obs_masks * obs
    obs = obs.reshape(obs_num, -1, 1)

    optim = torch.optim.LBFGS(params=[x_analysis],
                            lr=1e0,
                            max_iter=50,
                            max_eval=-1,
                            tolerance_grad=1e-7,
                            tolerance_change=1e-09,
                            history_size=100,
                            line_search_fn='strong_wolfe')
    
    def closure():
        loss_back = (x_analysis - x_init).T@B_inv@(x_analysis - x_init) 
        x_y = (obs_masks[0] * x_analysis.reshape(shape)).reshape(vec_shape)
        loss_obs = (x_y - obs[0]).T@R_inv@(x_y - obs[0])
        for i in range(daw//dtmodel-1):
            if i == 0:
                x = model(torch.unsqueeze(x_analysis.reshape(shape), dim=0)).reshape(x_analysis.shape)
            else:
                x = model(torch.unsqueeze(x.reshape(shape), dim=0)).reshape(x_analysis.shape)
            if ((i+1)*dtmodel) % dt_obs == 0:
                x_y = (obs_masks[(i+1)*dtmodel//dt_obs] * x.reshape(shape)).reshape(vec_shape)
                loss_obs += (x_y - obs[(i+1)*dtmodel//dt_obs]).T@R_inv@(x_y - obs[(i+1)*dtmodel//dt_obs])
        loss = 0.5*loss_back + 0.5*loss_obs
        loss.backward()
        return loss

    for nIters in range(maxIter):
        optim.step(closure)
        optim.zero_grad()
        model.zero_grad()
    
    del optim
    del obs
    torch.cuda.empty_cache() 
    x_analysis = x_analysis.detach()
    torch.set_grad_enabled(False)
    return x_analysis.reshape(shape)