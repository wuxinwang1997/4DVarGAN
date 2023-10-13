import torch
import numpy as np
import time
import xarray as xr
from src.da_methods.var4d import Solve_Var4D
# from src.da_methods.var3d import Solve_Var3D
from src.da_methods.spnekf import SPENKF
from src.utils.tools import gaussian_perturb
from src.utils.score import compute_weighted_rmse, compute_weighted_mae, compute_weighted_acc

def autoregressive_inference_background(ic, mean, std, valid_data_full, model, dtmodel, dt, prediction_length, device):
    ic = int(ic)
    model = model.to(device)
    dt = dt
    prediction_length = int(prediction_length)

    seq_pred = torch.zeros((prediction_length // dt, 1, 1, 64, 128)).to(device, dtype=torch.float)

    init_data = valid_data_full['z'][ic-prediction_length:ic:dt].values #extract valid data from first year
    # standardize
    init_data = (init_data - mean)/std
    init_data = torch.as_tensor(init_data).to(device, dtype=torch.float)

    with torch.no_grad():
        for ini in range(init_data.shape[0]):
            for i in range(prediction_length // dt - ini):
                for j in range(dt // dtmodel):
                    # 从ic开始
                    if j==0: #start of sequence                        
                        future_pred = model(torch.unsqueeze(torch.unsqueeze(init_data[ini], dim=0), dim=0))
                    else:
                        future_pred = model(future_pred) #autoregressive step

            seq_pred[ini] = torch.unsqueeze(future_pred, dim=2)
        
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['init_time', 'time', 'level', 'lat', 'lon'],
            coords={
                'init_time': np.arange(-prediction_length, 0, dt),
                'time': valid_data_full.time.values[ic:ic+1], 
                'level': [valid_data_full.level.values],
                'lat': valid_data_full.lat.values, 
                'lon': valid_data_full.lon.values
            },
            name='z'
        )

    del init_data, model, seq_pred, future_pred
    return pred_nc

# 单纯做逐dtmodel的预报
def autoregressive_inference_after_spin(ic_spin, ic, imgsize, mean, std, valid_data_full, ini_data, model, dtmodel, dt_obs, spin_length, prediction_length, clim, device):
    valid_loss = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length//dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length//dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    ini_data = (ini_data - mean) / std
    valid_data = valid_data_full['z'][ic:ic+prediction_length//dt_obs:dtmodel//dt_obs].values #extract valid data from first year
    # standardize
    valid_data = (valid_data - mean)/std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i==0: #start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first #extract history from 1st 
                seq_pred[0] = torch.from_numpy(ini_data).to(device, dtype=torch.float) if isinstance(ini_data, np.ndarray) else ini_data
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            else:
                # 存储下一时刻的真值
                if i < prediction_length//dtmodel-1:
                    future = valid_data[i+1]
                future_pred = model(future_pred) #autoregressive step
            
            if i < prediction_length//dtmodel-1: #not on the last step
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i], 
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            mae[i] = compute_weighted_mae(seq_pred[i], 
                                        seq_real[i], 
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            acc[i] = compute_weighted_acc(seq_pred[i]*std+mean, 
                                        seq_real[i]*std+mean, 
                                        clim,
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))
            
                         
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[1:] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(spin_length+dtmodel, spin_length+prediction_length, dtmodel),
                'init_time': valid_data_full.time.values[ic_spin:ic_spin + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[1:], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[1:], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[1:], axis=0)
    return pred_nc, np_valid_loss, np_acc, np_mae

def autoregressive_inference_4dvar(ic, start_id, ai_spinup_length, imgsize, out_iter, mean, std, valid_data_full,
                                       ini_data, obs, model, da_pred_model, dtmodel, dt_da_pred, daw, dt_obs, B_inv,
                                       R_inv, prediction_length, init_length, obs_masks, clim, device):
    ic_obs = ic-start_id
    model = model.to(device)
    daw_dt = daw // dtmodel

    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length//dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length//dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year

    # standardize
    valid_data = (valid_data - mean)/std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    da_time = 0
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i==0: #start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first #extract history from 1st 
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_xb[0] = seq_pred[0]
                seq_pred[0] = Solve_Var4D(seq_pred[0],
                                                torch.from_numpy(B_inv).to(device, dtype=torch.float), 
                                                torch.from_numpy(R_inv).to(device, dtype=torch.float), 
                                                out_iter, # 外循环迭代次数
                                                da_pred_model,
                                                dt_da_pred,
                                                torch.from_numpy(obs[ic_obs:ic_obs+daw//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测
                                                dt_obs,
                                                torch.from_numpy(obs_masks[ic_obs:ic_obs+daw//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测算子
                                                daw)
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length//dtmodel-1:
                    future = valid_data[i+1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        # 存储背景场
                        seq_xb[int((i-(init_length//dtmodel))//daw_dt)] = seq_pred[i]
                        # 调用4DVar进行同化
                        start_time = time.time()
                        seq_pred[i] = Solve_Var4D(seq_pred[i],
                                                torch.from_numpy(B_inv).to(device, dtype=torch.float), 
                                                torch.from_numpy(R_inv).to(device, dtype=torch.float), 
                                                out_iter, # 外循环迭代次数
                                                da_pred_model,
                                                dt_da_pred,
                                                torch.from_numpy(obs[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测
                                                dt_obs,
                                                torch.from_numpy(obs_masks[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测算子
                                                daw)
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time
                        
                future_pred = model(future_pred)
                
            if i < prediction_length//dtmodel-1: #not on the last step
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i], 
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            mae[i] = compute_weighted_mae(seq_pred[i], 
                                        seq_real[i], 
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            acc[i] = compute_weighted_acc(seq_pred[i]*std+mean,
                                        seq_real[i]*std+mean, 
                                        clim,
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time


def autoregressive_inference_3dvar(ic, start_id, ai_spinup_length, imgsize, out_iter, mean, std, valid_data_full,
                                   ini_data, obs, model, dtmodel, dt_da_pred, daw, dt_obs, B_inv,
                                   R_inv, prediction_length, init_length, obs_masks, clim, device):
    ic_obs = ic - start_id
    model = model.to(device)
    daw_dt = daw // dtmodel

    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (
                prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year

    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_pred[i] = Solve_Var3D(seq_pred[0],
                                        torch.from_numpy(B_inv).to(device, dtype=torch.float),
                                        torch.from_numpy(R_inv).to(device, dtype=torch.float),
                                        out_iter,  # 外循环迭代次数
                                        torch.from_numpy(obs[ic_obs]).to(device, dtype=torch.float),  # 引入对应的观测
                                        torch.from_numpy(obs_masks[ic_obs]).to(device, dtype=torch.float),  # 引入对应的观测算子
                                        )
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        # 存储背景场
                        seq_xb[int((i-(init_length//dtmodel)) // daw_dt) - 1] = seq_pred[i]
                        # 调用4DVar进行同化
                        start_time = time.time()
                        seq_pred[i] = Solve_Var3D(seq_pred[i],
                                                  torch.from_numpy(B_inv).to(device, dtype=torch.float),
                                                  torch.from_numpy(R_inv).to(device, dtype=torch.float),
                                                  out_iter,  # 外循环迭代次数
                                                  torch.from_numpy(obs[ic_obs + i * dtmodel // dt_obs]).to(device, dtype=torch.float),  # 引入对应的观测
                                                  torch.from_numpy(obs_masks[ic_obs + i * dtmodel // dt_obs]).to(device, dtype=torch.float),  # 引入对应的观测算子
                                                  )
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device,
                                                                                                     dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_ai(ic, imgsize, ai_spinup_length, ensemble, start_id, mean, std, valid_data_full, ini_data,
                                  obs, model, damodel, dtmodel, daw, dt_obs, init_length, prediction_length,
                                  obs_masks, clim, device):
    ic_obs = ic + ai_spinup_length // dt_obs - start_id
    model = model.to(device)
    # model.train()
    # damodel.train()
    daw_dt = daw // dtmodel
    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    xas = torch.zeros((ensemble, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year
    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs]), dim=0).to(device, dtype=torch.float)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs]), dim=0).to(device, dtype=torch.float)
                input_obs = input_obs * input_mask + (1 - input_mask) * torch.unsqueeze(seq_pred[0], dim=0)
                seq_pred[0] = damodel(torch.concat([torch.unsqueeze(seq_pred[0], dim=0), input_obs], dim=1))
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        seq_xb[int((i-(init_length//dtmodel)) // daw_dt)] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs + i * dtmodel // dt_obs]), dim=0).to(
                                        device, dtype=torch.float)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs + i * dtmodel // dt_obs]),
                                                    dim=0).to(device, dtype=torch.float)
                        input_obs = input_obs * input_mask + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        if ensemble > 1:
                            for j in range(ensemble):
                                xas[j] = damodel(
                                    torch.concat([torch.unsqueeze(seq_pred[i], dim=0), input_obs], dim=1)
                                )
                            seq_pred[i] = torch.mean(xas, dim=0)
                        else:
                            seq_pred[i] = damodel(
                                    torch.concat([torch.unsqueeze(seq_pred[i], dim=0), input_obs], dim=1)
                                )
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_cyclegan(ic, imgsize, ai_spinup_length, ensemble, start_id, mean, std, valid_data_full, ini_data,
                                  obs, model, damodel, dtmodel, daw, dt_obs, init_length, prediction_length,
                                  obs_masks, clim, device):
    ic_obs = ic + ai_spinup_length // dt_obs - start_id
    model = model.to(device)
    # model.train()
    # damodel.train()
    daw_dt = daw // dtmodel
    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    xas = torch.zeros((ensemble, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year
    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs:ic_obs+daw // dt_obs,0]), dim=0).to(device, dtype=torch.float)
                input_mask = torch.unsqueeze(torch.from_numpy(
                            obs_masks[ic_obs:ic_obs + + daw // dt_obs,0]), dim=0).to(device, dtype=torch.float)
                input_obs = input_obs * input_mask
                input_obs = torch.concat([input_obs, input_mask], dim=1)
                seq_pred[0] = damodel(torch.concat([torch.unsqueeze(seq_pred[0], dim=0), input_obs], dim=1))
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        seq_xb[int((i-(init_length//dtmodel)) // daw_dt)] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs + i * dtmodel // dt_obs]), dim=0).to(
                                        device, dtype=torch.float)
                        # input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs + i * dtmodel // dt_obs]),
                        #                             dim=0).to(device, dtype=torch.float)
                        # input_obs = input_obs * input_mask #+ (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        # # input_obs = (input_obs - torch.unsqueeze(seq_pred[i], dim=0)) * input_mask #  + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        # input_obs = torch.concat([input_obs, input_mask], dim=1)
                        input_obs = torch.unsqueeze(torch.from_numpy(
                            obs[ic_obs + i * dtmodel // dt_obs:ic_obs + (i * dtmodel + daw) // dt_obs,0]), dim=0).to(
                            device, dtype=torch.float)
                        input_mask = torch.unsqueeze(torch.from_numpy(
                            obs_masks[ic_obs + i * dtmodel // dt_obs:ic_obs + (i * dtmodel + daw) // dt_obs,0]),
                                                     dim=0).to(device, dtype=torch.float)
                        input_obs = input_obs * input_mask
                        input_obs = torch.concat([input_obs, input_mask], dim=1)
                        if ensemble > 1:
                            for j in range(ensemble):
                                xas[j] = damodel(
                                    torch.concat([torch.unsqueeze(seq_pred[i], dim=0), input_obs], dim=1)
                                )
                            seq_pred[i] = torch.mean(xas, dim=0)
                        else:
                            seq_pred[i] = damodel(
                                    torch.concat([torch.unsqueeze(seq_pred[i], dim=0), input_obs], dim=1)
                                )
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_4dvarnet(ic, imgsize, ai_spinup_length, ensemble, start_id, mean, std, valid_data_full, ini_data,
                                  obs, model, damodel, dtmodel, daw, dt_obs, init_length, prediction_length,
                                  obs_masks, clim, device):
    ic_obs = ic + ai_spinup_length // dt_obs - start_id
    model = model.to(device)
    # model.train()
    # damodel.train()
    daw_dt = daw // dtmodel
    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    xas = torch.zeros((ensemble, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year
    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs:ic_obs+daw//dt_obs,0]), dim=0).to(device, dtype=torch.float)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs:ic_obs+daw//dt_obs,0]),dim=0).to(device, dtype=torch.float)
                input_obs = input_obs * input_mask
                with torch.set_grad_enabled(True):
                    xb = torch.autograd.Variable(torch.unsqueeze(seq_pred[0], dim=0), requires_grad=True)
                    seq_pred[0], hidden_new, cell_new, normgrad = damodel(xb, input_obs, input_mask)
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        seq_xb[int((i-(init_length//dtmodel)) // daw_dt)] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs,0]), dim=0).to(
                                        device, dtype=torch.float)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs,0]),
                                                    dim=0).to(device, dtype=torch.float)
                        input_obs = input_obs * input_mask #+ (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        # input_obs = (input_obs - torch.unsqueeze(seq_pred[i], dim=0)) * input_mask #  + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        with torch.set_grad_enabled(True):
                            xb = torch.autograd.Variable(torch.unsqueeze(seq_pred[i], dim=0), requires_grad=True)
                            if ensemble > 1:
                                for j in range(ensemble):
                                    xas[j], hidden_new, cell_new, normgrad = damodel(xb, input_obs, input_mask)
                                seq_pred[i] = torch.mean(xas, dim=0)
                            else:
                                seq_pred[i], hidden_new, cell_new, normgrad = damodel(xb, input_obs, input_mask)
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_4dvargan(ic, imgsize, ai_spinup_length, ensemble, start_id, mean, std, valid_data_full, ini_data,
                                  obs, model, damodel, dtmodel, daw, dt_obs, init_length, prediction_length,
                                  obs_masks, clim, device):
    ic_obs = ic + ai_spinup_length // dt_obs - start_id
    model = model.to(device)
    # model.train()
    # damodel.train()
    daw_dt = daw // dtmodel
    prediction_length = prediction_length + init_length - ai_spinup_length
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int((prediction_length-init_length)//daw), 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    xas = torch.zeros((ensemble, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic + ai_spinup_length // dt_obs:ic + (prediction_length + ai_spinup_length) // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year
    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                seq_xb[0] = seq_pred[0]
                input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs:ic_obs+daw//dt_obs,0]), dim=0).to(device, dtype=torch.float)
                input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs:ic_obs+daw//dt_obs,0]),dim=0).to(device, dtype=torch.float)
                input_obs = input_obs * input_mask
                # input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs]), dim=0).to(device, dtype=torch.float)
                # input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs]), dim=0).to(device, dtype=torch.float)
                # input_obs = input_obs * input_mask + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                with torch.set_grad_enabled(True):
                    xb = torch.autograd.Variable(torch.unsqueeze(seq_pred[0], dim=0), requires_grad=True)
                    seq_pred[0] = damodel(xb, input_obs, input_mask)
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            elif i < init_length // dtmodel:
                future = valid_data[i + 1]
                future_pred = model(future_pred)
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    if (i-(init_length//dtmodel)) % daw_dt == 0:
                        seq_xb[int((i-(init_length//dtmodel)) // daw_dt)] = seq_pred[i]
                        start_time = time.time()
                        input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs,0]), dim=0).to(
                                        device, dtype=torch.float)
                        input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs+i*dtmodel//dt_obs:ic_obs+(i*dtmodel+daw)//dt_obs,0]),
                                                    dim=0).to(device, dtype=torch.float)
                        input_obs = input_obs * input_mask #+ (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        # input_obs = (input_obs - torch.unsqueeze(seq_pred[i], dim=0)) * input_mask #  + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        # input_obs = torch.unsqueeze(torch.from_numpy(obs[ic_obs + i * dtmodel // dt_obs]), dim=0).to(
                        #                 device, dtype=torch.float)
                        # input_mask = torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs + i * dtmodel // dt_obs]),
                        #                             dim=0).to(device, dtype=torch.float)
                        # input_obs = input_obs * input_mask + (1 - input_mask) * torch.unsqueeze(seq_pred[i], dim=0)
                        with torch.set_grad_enabled(True):
                            xb = torch.autograd.Variable(torch.unsqueeze(seq_pred[i], dim=0), requires_grad=True)
                            if ensemble > 1:
                                for j in range(ensemble):
                                    xas[j] = damodel(xb, input_obs, input_mask)
                                seq_pred[i] = torch.mean(xas, dim=0)
                            else:
                                seq_pred[i] = damodel(xb, input_obs, input_mask)
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length, prediction_length + ai_spinup_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(ai_spinup_length + init_length, prediction_length + ai_spinup_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_after_spin_cyclegan(ic_spin, ic, imgsize, mean, std, valid_data_full, ini_data, model, damodel, dtmodel,
                                        dt_obs, spin_length, prediction_length, clim, device):
    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, imgsize[0], imgsize[1])).to(device, dtype=torch.float)
    ini_data = (ini_data - mean) / std
    valid_data = valid_data_full['z'][
                 ic:ic + prediction_length // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year
    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = torch.from_numpy(ini_data).to(device, dtype=torch.float) if isinstance(ini_data,
                                                                                                     np.ndarray) else ini_data
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                future_pred = model(future_pred)  # autoregressive step
                future_pred = damodel(torch.concat([future_pred, future_pred], dim=1))

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device,
                                                                                                     dtype=float)) * std
            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[1:] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(spin_length + dtmodel, spin_length + prediction_length, dtmodel),
                'init_time': valid_data_full.time.values[ic_spin:ic_spin + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[1:], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[1:], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[1:], axis=0)
    return pred_nc, np_valid_loss, np_acc, np_mae

def autoregressive_inference_enkf(ic,
                                  start_id,
                                  out_iter,
                                  mean,
                                  std,
                                  valid_data_full,
                                  ini_data,
                                  obs,
                                  model,
                                  dtmodel,
                                  daw,
                                  dt_obs,
                                  N,
                                  P,
                                  Q,
                                  R,
                                  u_ensemble,
                                  prediction_length,
                                  obs_masks,
                                  clim,
                                  device):
    ic_obs = ic - start_id
    model = model.to(device)
    daw_dt = daw // dtmodel

    valid_loss = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length // dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length // dtmodel, 1, 32, 64)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length // dtmodel, 1, 32, 64)).to(device, dtype=torch.float)
    seq_xb = torch.zeros((int(prediction_length // daw) - 1, 1, 32, 64)).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][
                 ic:ic + prediction_length // dt_obs:dtmodel // dt_obs].values  # extract valid data from first year

    # standardize
    valid_data = (valid_data - mean) / std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    da_time = 0

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i == 0:  # start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first  # extract history from 1st
                seq_pred[0] = gaussian_perturb(torch.from_numpy(ini_data).to(device, dtype=torch.float), level=0)
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            else:
                # 存储下一时刻的真值
                if i < prediction_length // dtmodel - 1:
                    future = valid_data[i + 1]
                    # 提前一个时间段开始做集合预报
                    if (i+1) % daw_dt == 0:
                        # 存储背景场
                        seq_xb[int(i // daw_dt) - 1] = seq_pred[i]
                        # 调用4DVar进行同化
                        start_time = time.time()
                        seq_pred[i], P = SPENKF(seq_pred[i],
                                                N,
                                                P,
                                                Q,
                                                R,
                                                model,
                                                obs[ic_obs + i * dtmodel // dt_obs],  # 引入对应的观测
                                                obs_masks[ic_obs + i * dtmodel // dt_obs],  # 引入对应的观测算子
                                                u_ensemble,
                                                device)
                        future_pred = torch.unsqueeze(seq_pred[i], dim=0)
                        end_time = time.time()
                        da_time += end_time - start_time

                future_pred = model(future_pred)

            if i < prediction_length // dtmodel - 1:  # not on the last step
                seq_pred[i + 1] = future_pred
                seq_real[i + 1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i],
                                                  seq_real[i],
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device,
                                                                                                     dtype=float)) * std

            mae[i] = compute_weighted_mae(seq_pred[i],
                                          seq_real[i],
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std

            acc[i] = compute_weighted_acc(seq_pred[i] * std + mean,
                                          seq_real[i] * std + mean,
                                          clim,
                                          torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))

        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy()[:-1] * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(0, prediction_length - dtmodel, dtmodel),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
        xb_nc = xr.DataArray(
            seq_xb.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'init_time', 'lat', 'lon'],
            coords={
                'lead_time': np.arange(daw, prediction_length, daw),
                'init_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values,
                'lon': valid_data_full.lon.values
            },
            name='z'
        )
    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy()[:-1], axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy()[:-1], axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy()[:-1], axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, seq_xb, first
    return pred_nc, xb_nc, np_valid_loss, np_acc, np_mae, da_time

def autoregressive_inference_4dvar_midrange(ic, start_id, out_iter, mean, std, valid_data_full, init_data, obs, model, da_pred_model, dtmodel, dt_da_pred, daw, dt_obs, B_inv, R_inv, prediction_length, obs_masks, device):
    ic_obs = ic-start_id
    model = model.to(device)
    daw_dt = daw // dtmodel
    clim = torch.from_numpy(valid_data_full.mean('time')['z'].values).to(device, dtype=torch.float)
    
    valid_loss = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length//dtmodel, 1, 32, 64)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length//dtmodel, 1, 32, 64)).to(device, dtype=torch.float)

    valid_data = valid_data_full['z'][ic:(ic+prediction_length//dt_obs):dtmodel//dt_obs].values #extract valid data from first year
    # standardize
    valid_data = (valid_data - mean)/std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i==0: #start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first #extract history from 1st
                seq_pred[0] = Solve_Var4D(torch.from_numpy(init_data).to(device, dtype=torch.float),
                                        torch.from_numpy(B_inv).to(device, dtype=torch.float), 
                                        torch.from_numpy(R_inv).to(device, dtype=torch.float), 
                                        out_iter, # 外循环迭代次数
                                        da_pred_model,
                                        dt_da_pred,
                                        torch.from_numpy(obs[ic_obs:ic_obs+daw//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测
                                        dt_obs,
                                        torch.from_numpy(obs_masks[ic_obs:ic_obs+daw//dt_obs]).to(device, dtype=torch.float), # 引入对应的观测算子
                                        daw)
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            else:
                # 存储下一时刻的真值
                if i < prediction_length//dtmodel-1:
                    future = valid_data[i+1]
                    
                future_pred = model(future_pred)
                
            if i < prediction_length//dtmodel-1: #not on the last step
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i], 
                                                  seq_real[i], 
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            mae[i] = compute_weighted_mae(seq_pred[i], 
                                        seq_real[i], 
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            acc[i] = compute_weighted_acc(seq_pred[i]*std+mean, 
                                        seq_real[i]*std+mean, 
                                        clim,
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))
            
                         
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                # 'lead_time': np.arange(0, prediction_length, dtmodel),
                'time': valid_data_full.time.values[ic:ic+prediction_length//dt_obs:dtmodel//dt_obs],
                'start_time': valid_data_full.time.values[ic:ic + 1],
                'lat': valid_data_full.lat.values, 
                'lon': valid_data_full.lon.values
            },
            name='z'
        )

    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy(), axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy(), axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy(), axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, first
    return pred_nc, np_valid_loss, np_acc, np_mae

def autoregressive_inference_cafnovae_midrange(ic, start_id, mean, std, valid_data_full, ini_data, obs, model, damodel, dtmodel, daw, dt_obs, prediction_length, obs_masks, device):
    ic_obs = ic-start_id
    model = model.to(device)
    daw_dt = daw // dtmodel
    clim = torch.from_numpy(valid_data_full.mean('time')['z'].values).to(device, dtype=torch.float)
    
    valid_loss = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    mae = torch.zeros((prediction_length//dtmodel, 1)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length//dtmodel, 1, 32, 64)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length//dtmodel, 1, 32, 64)).to(device, dtype=torch.float)
    
    valid_data = valid_data_full['z'][ic:(ic+prediction_length//dt_obs):(dtmodel//dt_obs)].values #extract valid data from first year
    # standardize
    valid_data = (valid_data - mean)/std
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            # 从ic开始
            if i==0: #start of sequence
                first = valid_data[0]
                future = valid_data[1]
                seq_real[0] = first #extract history from 1st 
                seq_pred[0], _, __ = damodel(
                            torch.unsqueeze(torch.from_numpy(ini_data), dim=0).to(device, dtype=torch.float),
                            torch.unsqueeze(torch.from_numpy(obs[ic_obs]), dim=0).to(device, dtype=torch.float),
                            torch.unsqueeze(torch.from_numpy(obs_masks[ic_obs]), dim=0).to(device, dtype=torch.float),
                        )
                future_pred = model(torch.unsqueeze(seq_pred[0], dim=0))
            else:
                # 存储下一时刻的真值
                if i < prediction_length//dtmodel-1:
                    future = valid_data[i+1]
                # 如果到了ic+3dt，那么准备做集合预报，并在第ic+4dt时刻做同化         
                future_pred = model(future_pred)
                
            if i < prediction_length//dtmodel-1: #not on the last step
                seq_pred[i+1] = future_pred
                seq_real[i+1] = future

            valid_loss[i] = compute_weighted_rmse(seq_pred[i], 
                                                  seq_real[i], 
                                                  torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            mae[i] = compute_weighted_mae(seq_pred[i], 
                                        seq_real[i], 
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float)) * std
            
            acc[i] = compute_weighted_acc(seq_pred[i]*std+mean, 
                                        seq_real[i]*std+mean, 
                                        clim,
                                        torch.from_numpy(valid_data_full['lat'].values).to(device, dtype=float))
            
                         
        pred_nc = xr.DataArray(
            seq_pred.cpu().detach().numpy() * std + mean,
            dims=['lead_time', 'time', 'lat', 'lon'],
            coords={
                'time': valid_data_full.time.values[ic:ic+prediction_length//dt_obs:dtmodel//dt_obs],
                'start_time': valid_data_full.time.values[ic:ic+1], #np.arange(0, prediction_length, dtmodel),
                'lat': valid_data_full.lat.values, 
                'lon': valid_data_full.lon.values
            },
            name='z'
        )

    np_valid_loss = np.expand_dims(valid_loss.cpu().numpy(), axis=0)
    np_acc = np.expand_dims(acc.cpu().numpy(), axis=0)
    np_mae = np.expand_dims(mae.cpu().numpy(), axis=0)
    del valid_loss, acc, future, future_pred, valid_data, valid_data_full, seq_pred, seq_real, first
    return pred_nc, np_valid_loss, np_acc, np_mae