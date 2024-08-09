import sys
sys.path.append(".")
import os
from pathlib import Path
import pickle
import numpy as np
import torch
from src.models.forecast.forecast_module import ForecastLitModule
from src.evaluate.inference import autoregressive_inference
import matplotlib as plt
import argparse
import xarray as xr
from src.utils.data_utils import NAME_TO_VAR


def forecast_inference(
        era5_dir,
        ckpt_dir,
        output_dir,
        var,
        forecast_hours,
        decorrelation_hours,
        mode,
        device):

    decorrelation_length = decorrelation_hours // 6

    forecast_model = ForecastLitModule.load_from_checkpoint(f"{ckpt_dir}/afnonet_{NAME_TO_VAR[var]}{var[-3:]}_0714.ckpt")
    forecast_net = forecast_model.net.to(device).eval()

    mult = forecast_model.mult
    clim = forecast_model.clims[2]

    mean = np.load(os.path.join(f"{era5_dir}/{var}_5.625deg", 'normalize_mean.npy'))
    std = np.load(os.path.join(f"{era5_dir}/{var}_5.625deg", 'normalize_std.npy'))

    era5 = xr.open_mfdataset(f"{era5_dir}/{var}_5.625deg/{mode}/*.nc", combine='by_coords')

    # 取初始场
    n_samples = era5[NAME_TO_VAR[var]].shape[0] - forecast_hours
    stop = n_samples
    ics = np.arange(0, stop, decorrelation_length)

    val_rmse, val_acc, val_mae = [], [], []

    fcs = []
    for i, ic in enumerate(ics):
        pred_nc, seq_rmse, seq_acc, seq_mae = autoregressive_inference(ic, mean, std, era5, 
                                                                        forecast_net, mult, clim, 
                                                                        forecast_hours, 6, var, device)

        fcs.append(pred_nc)
        if i == 0:
            val_rmse = seq_rmse
            val_acc = seq_acc
            val_mae = seq_mae
        else:
            val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
            val_acc = np.concatenate((val_acc, seq_acc), axis=0)
            val_mae = np.concatenate((val_mae, seq_mae), axis=0)
        
        print(f'Exp {i+1}/{len(ics)} end.')

    for i in range(val_rmse.shape[-1]):
        print(f"RMSE of {var} is: {np.mean(val_rmse, axis=0)[:, i]}")
        print(f"ACC of {var} is: {np.mean(val_acc, axis=0)[:, i]}")
        print(f"MAE of {var} is: {np.mean(val_mae, axis=0)[:, i]}")

    fc_iter = xr.merge(fcs)
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

    fc_iter.to_netcdf(f'{output_dir}/fc_iter_afnonet.nc')
    print(f'{output_dir}/fc_iter_afnonet.nc saved!')
    xr_rmse.to_netcdf(f'{output_dir}/rmse_afnonet.nc')
    print(f'{output_dir}/rmse_afnonet.nc saved!')
    xr_acc.to_netcdf(f'{output_dir}/acc_afnonet.nc')
    print(f'{output_dir}/acc_afnonet.nc saved!')
    xr_mae.to_netcdf(f'{output_dir}/mae_afnonet.nc')
    print(f'{output_dir}/mae_afnonet.nc saved!')

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--era5_dir',
        type=str,
        help='era5 data directory',
        default='../../data/era5'
    )

    parser.add_argument(
        '--ckpt_dir',
        type=str,
        help='ckpt directory',
        default='../ckpt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='output directory',
        default='../../data/results'
    )

    parser.add_argument(
        '--var',
        type=str,
        help='variable name',
        default='geopotential_500'
    )

    parser.add_argument(
        '--forecast_hours',
        type=int,
        help='length of the forecasting hours [h]',
        default=168
    )

    parser.add_argument(
        '--decorrelation_hours',
        type=int,
        help='decoorelation between each initial time [h]',
        default=168
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='mode of data',
        default='test'
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    era5_dir = args.era5_dir
    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    var = args.var
    forecast_hours = args.forecast_hours
    decorrelation_hours = args.decorrelation_hours
    mode = args.mode
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    forecast_inference(
        era5_dir,
        ckpt_dir,
        output_dir,
        var,
        forecast_hours,
        decorrelation_hours,
        mode,
        device)