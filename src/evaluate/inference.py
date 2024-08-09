import torch
import numpy as np
import time
import xarray as xr
import torch
from src.utils.weighted_acc_rmse import weighted_acc_torch, weighted_rmse_torch, weighted_mae_torch
from src.utils.data_utils import NAME_TO_VAR
from src.da_method.var4d import Solve_Var4D

def autoregressive_inference(ic, mean, std, era5, 
                            forecast_net, mult, clim, 
                            forecast_hours, dt, variable, device):
    ic = int(ic)
    prediction_length = int(forecast_hours // dt)
    clim = clim.to(device, dtype=torch.float32)
    mult = mult

    seq_pred = torch.zeros((1 + prediction_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((1 + prediction_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_rmse = torch.zeros((1 + prediction_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((1 + prediction_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((1 + prediction_length, 1)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+(forecast_hours+1)*dt+1:dt].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)
    init_data = valid_data[0:1]

    with torch.no_grad():
        for i in range(1 + prediction_length):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = init_data
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(valid_data[i:i+1])
                # Switch the input back to the stored input
                seq_pred[i:i+1] = forecast_net(seq_pred[i-1:i])

                seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
                seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)
                seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])

        seq_pred = seq_pred.cpu().detach().numpy()
        seq_real = seq_real.cpu().detach().numpy()
        seq_rmse = seq_rmse.cpu().detach().numpy()
        seq_acc = seq_acc.cpu().detach().numpy()
        seq_mae = seq_mae.cpu().detach().numpy()

        pred_nc = xr.DataArray(
            seq_pred * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, forecast_hours + dt, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

        return pred_nc, np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0), np.expand_dims(seq_mae, 0)

def autoregressive_inference_background(ic, mean, std, nc_data_full, module, prediction_length, dt, variable, device):
    ic = int(ic)

    seq_pred = torch.zeros((prediction_length // dt, 1, 1, 32, 64)).to(device, dtype=torch.float32)
    init_data = nc_data_full[NAME_TO_VAR[variable]][ic - prediction_length : ic : dt].values
    init_data = (init_data - mean) / std
    # standardize
    init_data = torch.as_tensor(init_data).to(device, dtype=torch.float32)

    with torch.no_grad():
        for ini in range(init_data.shape[0]):
            for i in range(prediction_length // dt - ini):
                for j in range(dt // 6):
                    # 从ic开始
                    if j == 0:                  
                        future_pred = module(torch.unsqueeze(torch.unsqueeze(init_data[ini], dim=0), dim=0))
                    else:
                        future_pred = module(future_pred)

            seq_pred[ini] = torch.unsqueeze(future_pred, dim=2)

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['init_time', 'time', "level", 'lat', 'lon'],
            coords={
                'init_time': np.arange(-prediction_length, 0, dt),
                'time': nc_data_full.time.values[ic:ic+1], 
                "level": [nc_data_full.level.values],
                'lat': nc_data_full.lat.values, 
                'lon': nc_data_full.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

    del init_data, module, seq_pred, future_pred
    return pred_nc

def autoregressive_inference_4dvar(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                  pred_model, mult, clim, B_inv, R_inv, maxIter,
                                  cycle_hours, daw, dt, variable, device):
    ic = int(ic)
    cycle_length = cycle_hours // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_xb = torch.zeros((cycle_hours // daw, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+cycle_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    da_time = 0
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.from_numpy((obs[ic:ic+3]- mean) / std).to(device, dtype=torch.float32)
                input_mask = torch.from_numpy(obs_mask[ic:ic+3]).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                seq_pred[0] = Solve_Var4D(seq_pred[0:1], B_inv, R_inv, maxIter, pred_model, dt, input_obs, input_mask, daw)
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < cycle_length - 1:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        seq_xb[i // daw_dt] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.from_numpy((obs[ic+i:ic+i+3]- mean) / std).to(device, dtype=torch.float32)
                        input_mask = torch.from_numpy(obs_mask[ic+i:ic+i+3]).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        seq_pred[i] = Solve_Var4D(seq_pred[i:i+1], B_inv, R_inv, maxIter, pred_model, dt, input_obs, input_mask, daw)
                        future_pred = pred_model(seq_pred[i:i+1])
                        end_time = time.time()
                        da_time += end_time - start_time
                future_pred = pred_model(future_pred)
        
            if i < cycle_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
        
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, daw),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

    return pred_nc, xb_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0), da_time


def autoregressive_medium_forecast_4dvar(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                  pred_model, mult, clim, B_inv, R_inv, maxIter,
                                  spinup_hours, forecast_hours, daw, dt, variable, device):
    ic = int(ic)
    spinup_length = spinup_hours // dt
    exp_length = (spinup_hours + forecast_hours) // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+exp_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                input_obs = torch.from_numpy((obs[ic:ic+3]- mean) / std).to(device, dtype=torch.float32)
                input_mask = torch.from_numpy(obs_mask[ic:ic+3]).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                seq_pred[0] = Solve_Var4D(seq_pred[0:1], B_inv, R_inv, maxIter, pred_model, dt, input_obs, input_mask, daw)
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < spinup_length - 1:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        input_obs = torch.from_numpy((obs[ic+i:ic+i+3]- mean) / std).to(device, dtype=torch.float32)
                        input_mask = torch.from_numpy(obs_mask[ic+i:ic+i+3]).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        seq_pred[i] = Solve_Var4D(seq_pred[i:i+1], B_inv, R_inv, maxIter, pred_model, dt, input_obs, input_mask, daw)
                        future_pred = pred_model(seq_pred[i:i+1])
                future_pred = pred_model(future_pred)
        
            if i < exp_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
        
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, spinup_hours + forecast_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
    
    return pred_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0)


def autoregressive_inference_aida(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                  pred_model, mult, clim, da_model, 
                                  cycle_hours, daw, dt, variable, device):
    ic = int(ic)
    cycle_length = cycle_hours // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_xb = torch.zeros((cycle_hours // daw, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+cycle_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    da_time = 0
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy((obs[ic:ic+1]- mean) / std), dim=0).to(device, dtype=torch.float32)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic:ic+1]), dim=0).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                seq_pred[0] = da_model(seq_pred[0:1], input_obs, input_mask)
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < cycle_length - 1:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        seq_xb[i // daw_dt] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.unsqueeze(torch.from_numpy((obs[ic+i:ic+i+1]- mean) / std), dim=0).to(device, dtype=torch.float32)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic+i:ic+i+1]), dim=0).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        seq_pred[i] = da_model(seq_pred[i:i+1], input_obs, input_mask)
                        future_pred = pred_model(seq_pred[i:i+1])
                        end_time = time.time()
                        da_time += end_time - start_time
                future_pred = pred_model(future_pred)
        
            if i < cycle_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
            
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, daw),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

    return pred_nc, xb_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0), da_time

def autoregressive_medium_forecast_aida(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                  pred_model, mult, clim, da_model, 
                                  spinup_hours, forecast_hours, daw, dt, variable, device):
    ic = int(ic)
    spinup_length = spinup_hours // dt
    exp_length = (spinup_hours + forecast_hours) // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+exp_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                input_obs = torch.unsqueeze(torch.from_numpy((obs[ic:ic+1]- mean) / std), dim=0).to(device, dtype=torch.float32)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic:ic+1]), dim=0).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                seq_pred[0] = da_model(seq_pred[0:1], input_obs, input_mask)
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < spinup_length:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        input_obs = torch.unsqueeze(torch.from_numpy((obs[ic+i:ic+i+1]- mean) / std), dim=0).to(device, dtype=torch.float32)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic+i:ic+i+1]), dim=0).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        seq_pred[i] = da_model(seq_pred[i:i+1], input_obs, input_mask)
                        future_pred = pred_model(seq_pred[i:i+1])
                future_pred = pred_model(future_pred)
        
            if i < exp_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
            
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, spinup_hours + forecast_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
    return pred_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0)



def autoregressive_inference_4dvarnn(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                    pred_model, mult, clim, da_model, 
                                    cycle_hours, daw, dt, variable, device):
    ic = int(ic)
    cycle_length = cycle_hours // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_xb = torch.zeros((cycle_hours // daw, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+cycle_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    da_time = 0
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy((obs[ic:ic+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic:ic+3]), dim=0).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                start_time = time.time()
                with torch.set_grad_enabled(True):
                    xb = torch.autograd.Variable(seq_pred[i:i+1], requires_grad=True)
                    seq_pred[i] = da_model(xb, input_obs, input_mask)
                end_time = time.time()
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < cycle_length - 1:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        seq_xb[i // daw_dt] = seq_pred[i]
                        input_obs = torch.unsqueeze(torch.from_numpy((obs[ic+i:ic+i+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic+i:ic+i+3]), dim=0).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        start_time = time.time()
                        with torch.set_grad_enabled(True):
                            xb = torch.autograd.Variable(seq_pred[i:i+1], requires_grad=True)
                            seq_pred[i] = da_model(xb, input_obs, input_mask)
                        end_time = time.time()
                        da_time += end_time - start_time
                    future_pred = pred_model(seq_pred[i:i+1])
        
            if i < cycle_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
            
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, daw),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

    return pred_nc, xb_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0), da_time


def autoregressive_medium_forecast_4dvarnn(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                    pred_model, mult, clim, da_model, 
                                    spinup_hours, forecast_hours, daw, dt, variable, device):
    ic = int(ic)
    spinup_length = spinup_hours // dt
    exp_length = (spinup_hours + forecast_hours) // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((exp_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((exp_length, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+exp_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                input_obs = torch.unsqueeze(torch.from_numpy((obs[ic:ic+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic:ic+3]), dim=0).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                with torch.set_grad_enabled(True):
                    xb = torch.autograd.Variable(seq_pred[i:i+1], requires_grad=True)
                    seq_pred[i] = da_model(xb, input_obs, input_mask)
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < spinup_length:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        input_obs = torch.unsqueeze(torch.from_numpy((obs[ic+i:ic+i+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic+i:ic+i+3]), dim=0).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        with torch.set_grad_enabled(True):
                            xb = torch.autograd.Variable(seq_pred[i:i+1], requires_grad=True)
                            seq_pred[i] = da_model(xb, input_obs, input_mask)
                future_pred = pred_model(seq_pred[i:i+1])
        
            if i < exp_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
            
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, spinup_hours + forecast_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
    return pred_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0)


def autoregressive_inference_cyclegan(ic, mean, std, era5, ini_data, obs, obs_mask, 
                                    pred_model, mult, clim, da_model, 
                                    cycle_hours, daw, dt, variable, device):
    ic = int(ic)
    cycle_length = cycle_hours // dt
    daw_dt = daw // dt
    seq_rmse = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_mae = torch.zeros((cycle_length, 1)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_pred = torch.zeros((cycle_length, 1, 32, 64)).to(device, dtype=torch.float32)
    seq_xb = torch.zeros((cycle_hours // daw, 1, 32, 64)).to(device, dtype=torch.float32)
    
    valid_data = era5[NAME_TO_VAR[variable]][ic:ic+cycle_length].values
    valid_data = (valid_data - mean) / std
    # standardize
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float32)

    da_time = 0
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first
                seq_pred[0] = torch.from_numpy((ini_data - mean) / std).to(device, dtype=torch.float32)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy((obs[ic:ic+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic:ic+3]), dim=0).to(device, dtype=torch.float32)
                input_obs = input_obs * input_mask
                start_time = time.time()
                seq_pred[i] = da_model(seq_pred[i:i+1], input_obs, input_mask)
                end_time = time.time()
                future_pred = pred_model(seq_pred[0:1])
            else:
                if i < cycle_length - 1:
                    future = valid_data[i + 1]
                    if i % daw_dt == 0:
                        seq_xb[i // daw_dt] = seq_pred[i]
                        input_obs = torch.unsqueeze(torch.from_numpy((obs[ic+i:ic+i+3]- mean) / std), dim=0).to(device, dtype=torch.float32)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_mask[ic+i:ic+i+3]), dim=0).to(device, dtype=torch.float32)
                        input_obs = input_obs * input_mask
                        start_time = time.time()
                        seq_pred[i] = da_model(seq_pred[i:i+1], input_obs, input_mask)
                        end_time = time.time()
                        da_time += end_time - start_time
                    future_pred = pred_model(seq_pred[i:i+1])
        
            if i < cycle_length - 1:
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future
            
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim.to(device, dtype=torch.float32), seq_pred[i:i+1] - clim.to(device, dtype=torch.float32))
            seq_mae[i:i + 1] = mult * weighted_mae_torch(seq_real[i:i+1], seq_pred[i:i+1])
            
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, dt),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )
        
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, cycle_hours, daw),
                'time': era5.time.values[ic:ic+1], 
                'lat': era5.lat.values, 
                'lon': era5.lon.values
            },
            name=NAME_TO_VAR[variable]
        )

    return pred_nc, xb_nc, np.expand_dims(seq_rmse.detach().cpu().numpy(), 0), np.expand_dims(seq_acc.detach().cpu().numpy(), 0), np.expand_dims(seq_mae.detach().cpu().numpy(), 0), da_time
