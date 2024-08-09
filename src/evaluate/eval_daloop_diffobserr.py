import sys
sys.path.append(".")
import os
from pathlib import Path
import pickle
import numpy as np
import torch
from src.models.forecast.forecast_module import ForecastLitModule
from src.models.assimilate.regressive_assim_module import AssimilateLitModule as RegressiveAssimModule
from src.models.assimilate.fdvarnet_assim_module import AssimilateLitModule as FdvarNetAssimModule
from src.models.assimilate.fdvarunet_assim_module import AssimilateLitModule as FdvarUNetAssimModule
from src.models.assimilate.fdvargan_assim_module import AssimilateLitModule as FdvarGANAssimModule
from src.models.assimilate.fdvarcyclegan_assim_module import AssimilateLitModule as FdvarCycleGANAssimModule
from src.models.assimilate.cyclegan_assim_module import AssimilateLitModule as CycleGANAssimModule
from src.evaluate.inference import autoregressive_inference, autoregressive_inference_aida, autoregressive_inference_4dvarnn, autoregressive_inference_4dvar, autoregressive_inference_cyclegan
from src.da_method.var4d import GC_1, GC_2
import matplotlib as plt
import argparse
import xarray as xr
from src.utils.data_utils import NAME_TO_VAR

def laplacian(psi, lats):
    lats = np.reshape(lats, (1, 1, lats.shape[0], 1))
    dx = lats[0, 0, 2, 0] - lats[0, 0, 0, 0]  # 经度步长
    # 计算经度方向的梯度（沿纬度变化）
    dpsidx = (psi[:, :, 1:-1, 2:] - psi[:, :, 1:-1, :-2]) / dx
    dpsidx = dpsidx / (111e3 * np.cos(np.deg2rad(lats[:,:,1:-1])))
    d2psidx2 = (dpsidx[:, :, 1:-1, 2:] - dpsidx[:, :, 1:-1, :-2]) / dx
    d2psidx2 = d2psidx2 / (111e3 * np.cos(np.deg2rad(lats[:,:,2:-2])))
        
    # 计算纬度方向的梯度（沿经度变化）
    dpsidy = (psi[:, :, 2:, 1:-1] - psi[:, :, :-2, 1:-1]) / dx
    dpsidy = dpsidy / 111e3
    d2psidy2 = (dpsidy[:, :, 2:, 1:-1] - dpsidx[:, :, :-2, 1:-1]) / dx
    d2psidy2 = d2psidy2 / 111e3

    return d2psidx2 + d2psidy2, dx

def daloop_inference(era5_dir,
                     xb_dir,
                     obs_dir,
                     obs_mask_dir,
                     ckpt_dir,
                     output_dir,
                     var,
                     init_time,
                     daw,
                     obs_partial,
                     obs_err,
                     obs_err_type,
                     maxIter,
                     inflation,
                     cycle_hours,
                     decorrelation_hours,
                     mode,
                     da_model,
                     device):

    decorrelation_length = decorrelation_hours // 6

    forecast_model = ForecastLitModule.load_from_checkpoint(f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
    forecast_net = forecast_model.net.to(device).eval()

    mult = forecast_model.mult
    clim = forecast_model.clims[2]
    
    if da_model in ["vit"]:
        assim_model = RegressiveAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt")
        assim_net = assim_model.net.to(device).eval()
    elif da_model in ["4dvarnet"]:
        assim_model = FdvarNetAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt",
                                                               pred_ckpt=f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
        assim_net = assim_model.net.to(device).eval()
    elif da_model in ["4dvarunet_woscale", "4dvarunet_wscale"]:
        assim_model = FdvarUNetAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt",
                                                                pred_ckpt=f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
        assim_net = assim_model.net.to(device).eval()
    elif da_model in ["4dvargan_woscale", "4dvargan_wscale"]:
        assim_model = FdvarGANAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt",
                                                               pred_ckpt=f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
        assim_net = assim_model.g_A2B.to(device).eval()
    elif da_model in ["4dvarcyclegan_woscale", "4dvarcyclegan_wscale", "4dvarcyclegan_scalein"]:
        assim_model = FdvarCycleGANAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt",
                                                                    pred_ckpt=f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
        assim_net = assim_model.g_A2B.to(device).eval()
    elif da_model in ["cyclegan_wscale"]:
        assim_model = CycleGANAssimModule.load_from_checkpoint(f"{ckpt_dir}/{da_model}_assim_{NAME_TO_VAR[var]}{var[-3:]}.ckpt",
                                                                pred_ckpt=f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
        assim_net = assim_model.g_A2B.to(device).eval()
    
    mean = np.load(os.path.join(f"{era5_dir}/{var}_5.625deg", 'normalize_mean.npy'))
    std = np.load(os.path.join(f"{era5_dir}/{var}_5.625deg", 'normalize_std.npy'))
    xb = xr.open_mfdataset(f"{xb_dir}/{mode}/*.nc", combine='by_coords')
    if da_model == "4dvar":
        # NMC方法计算B矩阵，并对B矩阵求逆
        pred_24 = xb.sel(init_time=-24)
        pred_48 = xb.sel(init_time=-48)
        diff = (pred_48 - pred_24)[NAME_TO_VAR[var]].values / std
        diff = np.reshape(diff, [diff.shape[0], -1])
        B = np.cov(diff.T)
        Nx = B.shape[0]
        # Create the correlation matrix corr_matrix
        # Create two grid martix to represent the indexs of the row and collumn, seperately
        row_indices, col_indices = np.meshgrid(np.arange(Nx), np.arange(Nx))
        # Computed th distance matrix
        dist_matrix = np.abs(row_indices - col_indices)
        # Compute the correlation matrix
        dist_matrix = dist_matrix // np.sqrt(Nx) + dist_matrix % np.sqrt(Nx)
        xb_value = xb.sel(init_time=-init_time)[NAME_TO_VAR[var]].values
        laplacian_pred, dx = laplacian(xb_value, xb["lat"].values)
        corr_len = np.sqrt(np.sqrt(8 * np.var(xb_value) / np.var(laplacian_pred)))
        dist_matrix = dist_matrix / (corr_len / (111e3 * dx / 2))
        corr_matrix = ((dist_matrix<1)*GC_1(dist_matrix)+(dist_matrix<2)*GC_2(dist_matrix/1)-(dist_matrix<1)*GC_2(dist_matrix))*(1-(dist_matrix>2))
        B = np.diag(np.diag(B)) * inflation + B * corr_matrix
        # B = inflation * np.eye(Nx) * np.max(np.diag(B))
        B_inv = np.linalg.inv(B)
        if obs_err_type == "ratio":
            obs_err = obs_err * mean / std
        elif obs_err_type == "constant":
            pass
        R_inv = 1 / (obs_err ** 2) * np.eye(Nx)

    obs = xr.open_mfdataset(f"{obs_dir}/{var}_5.625deg_obserr{obs_err}/{mode}/*.nc", combine='by_coords')[NAME_TO_VAR[var]].sel(time=xb["time"]).values
    obs_mask = xr.open_mfdataset(f"{obs_mask_dir}/{mode}/mask_paritial{obs_partial}*.nc")["mask"].sel(time=xb["time"]).values
    era5 = xr.open_mfdataset(f"{era5_dir}/{var}_5.625deg/{mode}/*.nc", combine='by_coords').sel(time=xb["time"])
    xb = xb.sel(init_time=-init_time)[NAME_TO_VAR[var]].values

    # 取初始场
    n_samples = xb.shape[0] - cycle_hours // 6
    stop = n_samples
    ics = np.arange(0, stop, decorrelation_length)

    val_rmse, val_acc, val_mae = [], [], []

    fcs, xbs, times4da = [], [], []
    for i, ic in enumerate(ics):
        if da_model in ["4dvar"]:
            pred_nc, xb_nc, seq_rmse, seq_acc, seq_mae, da_time = autoregressive_inference_4dvar(ic, mean, std, era5, xb[ic], obs, obs_mask, 
                                                                                    forecast_net, mult, clim, B_inv, R_inv, maxIter,
                                                                                    cycle_hours, daw, 6, var, device)
        elif da_model in ["vit"]:
            pred_nc, xb_nc, seq_rmse, seq_acc, seq_mae, da_time = autoregressive_inference_aida(ic, mean, std, era5, xb[ic], obs, obs_mask, 
                                                                                    forecast_net, mult, clim, assim_net,
                                                                                    cycle_hours, daw, 6, var, device)
        elif da_model in ["4dvarnet", "4dvarunet_woscale", "4dvarunet_wscale", "4dvargan_woscale", "4dvargan_wscale", "4dvarcyclegan_woscale", "4dvarcyclegan_wscale", "4dvarcyclegan_scalein"]:
            pred_nc, xb_nc, seq_rmse, seq_acc, seq_mae, da_time = autoregressive_inference_4dvarnn(ic, mean, std, era5, xb[ic], obs, obs_mask, 
                                                                                    forecast_net, mult, clim, assim_net,
                                                                                    cycle_hours, daw, 6, var, device)
        elif da_model in ["cyclegan_wscale"]:
            pred_nc, xb_nc, seq_rmse, seq_acc, seq_mae, da_time = autoregressive_inference_cyclegan(ic, mean, std, era5, xb[ic], obs, obs_mask, 
                                                                                    forecast_net, mult, clim, assim_net,
                                                                                    cycle_hours, daw, 6, var, device)
        
        fcs.append(pred_nc)
        xbs.append(xb_nc)
        times4da.append(da_time) 
        if i == 0:
            val_rmse = seq_rmse
            val_acc = seq_acc
            val_mae = seq_mae
        else:
            val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
            val_acc = np.concatenate((val_acc, seq_acc), axis=0)
            val_mae = np.concatenate((val_mae, seq_mae), axis=0)
        
        print(f'Exp {i+1}/{len(ics)} end.')
        print(f'Time for DA of {da_model} is {times4da[i]}[s]!')

    for i in range(val_rmse.shape[-1]):
        print(f"RMSE of {var} is: {np.nanmean(val_rmse, axis=0)[:, i]}")
        print(f"ACC of {var} is: {np.nanmean(val_acc, axis=0)[:, i]}")
        print(f"MAE of {var} is: {np.nanmean(val_mae, axis=0)[:, i]}")

    fc_iter = xr.merge(fcs)
    xb_iter = xr.merge(xbs)
    # 保存RMSE
    xr_rmse = [xr.DataArray(
                        np.mean(val_rmse, axis=0)[:, 0].astype(np.float32),
                        dims=['lead_time'],
                        coords={
                            'lead_time': fc_iter.lead_time.values,
                        },
                        name=NAME_TO_VAR[var]
                    )]
    xr_rmse = xr.merge(xr_rmse)
    # 保存ACC
    xr_acc = [xr.DataArray(
                        np.mean(val_acc, axis=0)[:, 0].astype(np.float32),
                        dims=['lead_time'],
                        coords={
                            'lead_time': fc_iter.lead_time.values,
                        },
                        name=NAME_TO_VAR[var]
                    )]
    xr_acc = xr.merge(xr_acc)
    # 保存MAE
    xr_mae = [xr.DataArray(
                        np.mean(val_mae, axis=0)[:, 0].astype(np.float32),
                        dims=['lead_time'],
                        coords={
                            'lead_time': fc_iter.lead_time.values,
                        },
                        name=NAME_TO_VAR[var]
                    )]
    xr_mae = xr.merge(xr_mae)

    xb_iter.to_netcdf(f'{output_dir}/xb_iter_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc')
    print(f'{output_dir}/xb_iter_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc saved!')
    fc_iter.to_netcdf(f'{output_dir}/fc_iter_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc')
    print(f'{output_dir}/fc_iter_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc saved!')
    xr_rmse.to_netcdf(f'{output_dir}/rmse_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc')
    print(f'{output_dir}/rmse_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc saved!')
    xr_acc.to_netcdf(f'{output_dir}/acc_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc')
    print(f'{output_dir}/acc_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc saved!')
    xr_mae.to_netcdf(f'{output_dir}/mae_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc')
    print(f'{output_dir}/mae_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.nc saved!')
    np.save(f'{output_dir}/da_time_{da_model.lower()}_obspartial{obs_partial}_obserr{obs_err}.npy', times4da)

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')
    
    parser.add_argument(
         '--era5_dir',
         type=str,
         help='path to era5 data',
         default='../../data/era5'
    )

    parser.add_argument(
        '--xb_dir',
        type=str,
        help='path to background fields',
        default='../../data/background'
    )

    parser.add_argument(
        '--obs_dir',
        type=str,
        help='path to observations',
        default='../../data/obs'
    )

    parser.add_argument(
        '--obs_mask_dir',
        type=str,
        help='path to observation mask',
        default='../../data/obs_mask'
    )

    parser.add_argument(
        '--ckpt_dir',
        type=str,
        help='path to checkpoints',
        default='../ckpt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path to output',
        default='../../data/results/assimilate'
    )

    parser.add_argument(
        '--var',
        type=str,
        help='variable name',
        default='geopotential_500'
    )
    
    parser.add_argument(
        '--init_time',
        type=int,
        help='lead time of the background field [h]',
        default=72
    )    

    parser.add_argument(
        '--daw',
        type=int,
        help='length of the assimilation window [h]',
        default=12
    )    
    
    parser.add_argument(
        '--obs_partial',
        type=float,
        help='partial of observational grids',
        default=0.2
    )  

    parser.add_argument(
        '--obs_err',
        type=float,
        help='error of observations',
        default=0.015
    )     

    parser.add_argument(
        '--obs_err_type',
        type=str,
        help='type of observation error',
        default="ratio"
    ) 
    
    parser.add_argument(
        '--cycle_hours',
        type=int,
        help='length of the assimilation cycle houes [h]',
        default=168
    )

    parser.add_argument(
        "--maxIter",
        type=int,
        help="maximum number of iterations",
        default=1,
    )

    parser.add_argument(
        "--inflation",
        type=float,
        help="inflation of the background error covariance",
        default=1.0
    )

    parser.add_argument(
        '--decorrelation_hours',
        type=int,
        help='decoorelation between each initial time [h]',
        default=120
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='mode of data',
        default='test'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='method used to do assimilation',
        default='vit'
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    era5_dir = args.era5_dir
    xb_dir = args.xb_dir
    obs_dir = args.obs_dir
    obs_mask_dir = args.obs_mask_dir
    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    var = args.var
    init_time = args.init_time
    daw = args.daw
    obs_partial = args.obs_partial
    obs_err=args.obs_err
    obs_err_type=args.obs_err_type
    cycle_hours = args.cycle_hours
    maxIter = args.maxIter
    inflation = args.inflation
    decorrelation_hours = args.decorrelation_hours
    mode = args.mode
    da_model = args.model_name
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    daloop_inference(era5_dir,
                     xb_dir,
                     obs_dir,
                     obs_mask_dir,
                     ckpt_dir,
                     output_dir,
                     var,
                     init_time,
                     daw,
                     obs_partial,
                     obs_err,
                     obs_err_type,
                     maxIter,
                     inflation,
                     cycle_hours,
                     decorrelation_hours,
                     mode,
                     da_model,
                     device)