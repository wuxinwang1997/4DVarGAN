# -*- coding=UTF-8 -*-
# -*- coding=UTF-8 -*-
# %%
# Depending on your combination of package versions, this can raise a lot of TF warnings... 
import numpy as np
# import matplotlib.pyplot as plt
import xarray as xr
import torch
import argparse
from pathlib import Path
import pickle
import sys
sys.path.append('.')
import time
from src.utils.score import *
# from src.utils.plot import plot_iter_result, plot_increment
from collections import OrderedDict
from src.inference.autoregressive_inference import autoregressive_inference_4dvar, autoregressive_inference_3dvar, autoregressive_inference_ai, autoregressive_inference_cyclegan, autoregressive_inference_4dvarnet, autoregressive_inference_4dvargan
from src.models.fourcast_module import FourCastNetLitModule # 预报模型
from src.models.map_module import MapLitModule
from src.models.cyclegan_module import CycleGANLitModule
from src.models.pred_cyclegan_module import CycleGANLitModule as FDVarCycleGANLitModule
from src.models.cyclegan_4d_module import FDCycleGANLitModule
from src.models.fdvarnet_module import FDVarNetLitModule
from src.models.fdvargan_module import CycleGANLitModule as FDVarGANModule
from src.utils.utils import str2bool
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def pred_da_loop_inference(data_dir, xb_dir, obs_dir, obs_mask_dir, pretrain_dir, damodel_dir, output_dir, init_time,
                           dtmodel, dt_obs, dt_da_pred, daw, obs_partial, obs_single, obserr_level, prediction_length,
                           decorrelation_time, out_iter, mode, da_method, ai_ensemble, resolution, device):
    # 读取均值、方差、经纬度网格数据
    with open(Path(f'{data_dir}') / f'scaler.pkl', 'rb') as f:
        item = pickle.load(f)
        lon = item['lon']
        lat = item['lat']
        mean = item['mean']
        std = item['std']
        f.close()
        
    # 读取真值
    valid = xr.open_mfdataset(f'{data_dir}/{mode}/*500*.nc', combine='by_coords')
    clim = torch.from_numpy(valid.mean('time')['z'].values).to(device, dtype=torch.float)
    # 读取观测数据
    obs = xr.open_mfdataset(f'{obs_dir}/{mode}/observations_*_err{obserr_level}.nc', combine='by_coords')
    obs = (obs - mean) / std
    obs_masks = xr.open_mfdataset(f'{obs_mask_dir}/{mode}/mask_observations_{int(100*obs_partial)}_*.nc', combine='by_coords')
    obs_masks = obs_masks.sel(time=obs['time'])['z'].values
    # 选取有观测对应时刻的真值
    valid = valid.sel(time=obs['time'])

    start_id = int((obs['time'][0]-valid["time"][0]).values.astype('timedelta64[h]') / np.timedelta64(1, 'h'))
    # 读取背景场数据
    xbs = xr.open_mfdataset(f'{xb_dir}/{mode}/background_predlen120_dtmodel6_year*.nc', combine='by_coords').sel(time=obs['time'])
    if da_method in ['4DVar', '3DVar']:
        # NMC方法计算B矩阵，并对B矩阵求逆
        pred_24 = xbs.sel(init_time=-24)
        pred_48 = xbs.sel(init_time=-48)
        diff = (pred_48 - pred_24)['z'].values / std
        diff = np.reshape(diff, [diff.shape[0], -1])
        B = np.cov(diff.T)
        B = np.diag(np.diag(B)) + np.max(np.diag(B))
        B_inv = np.linalg.inv(B)
    xbs = (xbs - mean) / std
    # 计算R矩阵并求逆
    noise = obserr_level * mean / std
    obs_value = obs['z'].values
    N = obs_value.shape[-1]*obs_value.shape[-2]
    R_inv = 1/(noise ** 2) * np.eye(N, N) # 观测误差协方差求逆
    
    # 提取用于作为初始场的xb
    xbs = xbs.sel(init_time=-init_time)
    
    # 加载预训练预报模型
    # 用于做预报的模型
    module = FourCastNetLitModule.load_from_checkpoint(f'{pretrain_dir}/fourcastnet_{resolution}deg_leadtime{dtmodel}h.ckpt')
    prednet = module.net.to(device).eval()
    # 用4DVar做同化的预报模型
    if da_method in ['4DVar', '3DVar']:
        da_module = FourCastNetLitModule.load_from_checkpoint(f'{pretrain_dir}/fourcastnet_{resolution}deg_leadtime{dt_da_pred}h.ckpt')
        da_prednet = da_module.net.to(device).eval()
    # 用ResNet做同化的预报模型
    elif da_method == 'ResNet':
        da_module = MapLitModule.load_from_checkpoint(f'{damodel_dir}/resnet_base.ckpt')
        da_model = da_module.net.to(device).eval()
    elif da_method == 'ViT':
        da_module = MapLitModule.load_from_checkpoint(f'{damodel_dir}/vit_base_0804_linear_embed.ckpt')
        da_model = da_module.net.to(device).eval()
    # 用LSTM做同化的预报模型
    elif da_method == 'LSTM':
        da_module = MapLitModule.load_from_checkpoint(f'{damodel_dir}/lstm_base_v1.ckpt')
        da_model = da_module.net.to(device).eval()
    # 用MLP做同化的预报模型
    elif da_method == 'MLP':
        da_module = MapLitModule.load_from_checkpoint(f'{damodel_dir}/mlp_base_2048.ckpt')
        da_model = da_module.net.to(device).eval()
    elif da_method == 'UNet':
        da_module = MapLitModule.load_from_checkpoint(f'{damodel_dir}/unet_base_0.0246.ckpt')
        da_model = da_module.net.to(device).eval()
    # 用CycleGAN做同化的预报模型
    elif da_method == 'CycleGAN':
        da_module = CycleGANLitModule.load_from_checkpoint(f'{damodel_dir}/cyclegan_lr1_5.ckpt')
        da_model = da_module.g_A2B.to(device).eval()
    elif da_method == '4DVarCycleGAN':
        da_module = FDVarCycleGANLitModule.load_from_checkpoint(f'{damodel_dir}/4dvarcyclegan_cyc10_idt5_4dvar50_obs20_0.0136.ckpt')
        da_model = da_module.g_A2B.to(device).eval()
    elif da_method == '4DCycleGAN':
        da_module = FDCycleGANLitModule.load_from_checkpoint(f'{damodel_dir}/unet_4dcyclegan_ymaskxb.ckpt')
        da_model = da_module.g_A2B.to(device).eval()
    elif da_method == '4DVarNet':
        da_module = FDVarNetLitModule.load_from_checkpoint(f'{damodel_dir}/4dvarnet_base_0804_.ckpt', ckpt=f'{pretrain_dir}/fourcastnet_{resolution}deg_leadtime{dtmodel}h.ckpt')
        da_model = da_module.net.to(device).eval()
    elif da_method == '4DVarGAN':
        da_module = FDVarGANModule.load_from_checkpoint(f'{damodel_dir}/4dvargan_pred60wL1_0806.ckpt', ckpt=f'{pretrain_dir}/fourcastnet_{resolution}deg_leadtime{dtmodel}h.ckpt')
        da_model = da_module.g_A2B.to(device).eval()

    # 取初始场
    prediction_length = 24 * prediction_length + 2 * dtmodel
    n_samples_per_year = len(valid['z'])
    n_samples = n_samples_per_year - prediction_length // dt_obs
    stop = n_samples
    ics = np.arange(start_id, stop, decorrelation_time // dt_obs)

    imgsize = [32, 64]
    fcs, xbs_ = [], []
    val_rmse, val_acc, val_mae, times4da = [], [], [], []
    for i, ic in enumerate(ics):
        print(f'Exp {i+1}/{len(ics)} start.')
        time4da = 0
        if da_method == '4DVar':
            fc, xb, rmse, acc, mae, da_time = autoregressive_inference_4dvar(ic,
                                                                            start_id,
                                                                            0,
                                                                            imgsize,
                                                                            out_iter,
                                                                            mean,
                                                                            std,
                                                                            valid,
                                                                            xbs['z'].values[ic],
                                                                            obs_value,
                                                                            prednet,
                                                                            da_prednet,
                                                                            dtmodel,
                                                                            dt_da_pred,
                                                                            daw,
                                                                            dt_obs,
                                                                            B_inv,
                                                                            R_inv,
                                                                            prediction_length,
                                                                            0,
                                                                            obs_masks,
                                                                            clim,
                                                                            device)
        elif da_method == '3DVar':
            fc, xb, rmse, acc, mae, da_time = autoregressive_inference_3dvar(ic,
                                                                            start_id,
                                                                            0,
                                                                            imgsize,
                                                                            out_iter,
                                                                            mean,
                                                                            std,
                                                                            valid,
                                                                            xbs['z'].values[ic],
                                                                            obs_value,
                                                                            prednet,
                                                                            dtmodel,
                                                                            dt_da_pred,
                                                                            daw,
                                                                            dt_obs,
                                                                            B_inv,
                                                                            R_inv,
                                                                            prediction_length,
                                                                            0,
                                                                            obs_masks,
                                                                            clim,
                                                                            device)

        elif da_method in ['ResNet', 'LSTM', 'MLP', 'ViT', 'UNet', 'CycleGAN']:
            fc, xb, rmse, acc, mae, da_time = autoregressive_inference_ai(ic,
                                                                        imgsize,
                                                                        0,
                                                                        ai_ensemble,
                                                                        start_id,
                                                                        mean,
                                                                        std,
                                                                        valid,
                                                                        xbs['z'].values[ic],
                                                                        obs['z'].values,
                                                                        prednet,
                                                                        da_model,
                                                                        dtmodel,
                                                                        daw,
                                                                        dt_obs,
                                                                        0,
                                                                        prediction_length,
                                                                        obs_masks,
                                                                        clim,
                                                                        device)
        elif da_method in ['4DVarGAN']:
            fc, xb, rmse, acc, mae, da_time = autoregressive_inference_4dvargan(ic,
                                                                            imgsize,
                                                                            0,
                                                                            ai_ensemble,
                                                                            start_id,
                                                                            mean,
                                                                            std,
                                                                            valid,
                                                                            xbs['z'].values[ic],
                                                                            obs['z'].values,
                                                                            prednet,
                                                                            da_model,
                                                                            dtmodel,
                                                                            daw,
                                                                            dt_obs,
                                                                            0,
                                                                            prediction_length,
                                                                            obs_masks,
                                                                            clim,
                                                                            device)
        elif da_method in ['4DVarNet']:
            fc, xb, rmse, acc, mae, da_time = autoregressive_inference_4dvarnet(ic,
                                                                        imgsize,
                                                                        0,
                                                                        ai_ensemble,
                                                                        start_id,
                                                                        mean,
                                                                        std,
                                                                        valid,
                                                                        xbs['z'].values[ic],
                                                                        obs['z'].values,
                                                                        prednet,
                                                                        da_model,
                                                                        dtmodel,
                                                                        daw,
                                                                        dt_obs,
                                                                        0,
                                                                        prediction_length,
                                                                        obs_masks,
                                                                        clim,
                                                                        device)
        fcs.append(fc)
        xbs_.append(xb)
        val_rmse.append(rmse)
        val_acc.append(acc)
        val_mae.append(mae)
        time4da += da_time
        times4da.append(time4da)
        del fc, xb, rmse, acc, mae, da_time
        print(f'Exp {i+1}/{len(ics)} end.')
        print(f'Time for DA of {da_method} is {time4da}[s]!')
    
    # 对每一组实验进行统计
    fc_iter = xr.merge(fcs)
    xb_iter = xr.merge(xbs_)
    val_rmse = np.mean(np.concatenate(val_rmse, 0), axis=0)
    val_acc = np.mean(np.concatenate(val_acc, 0), axis=0)
    val_mae = np.mean(np.concatenate(val_mae, 0), axis=0)
    # 保存RMSE
    xr_rmse = [xr.DataArray(
                        val_rmse[:,0],
                        dims=['Lead Time'],
                        coords={
                            'Lead Time': fc_iter.lead_time.values,
                        },
                        name='z'
                    )]
    xr_rmse = xr.merge(xr_rmse)
    # 保存ACC
    xr_acc = [xr.DataArray(
                        val_acc[:,0],
                        dims=['Lead Time'],
                        coords={
                            'Lead Time': fc_iter.lead_time.values,
                        },
                        name='z'
                    )]
    xr_acc = xr.merge(xr_acc)
    # 保存MAE
    xr_mae = [xr.DataArray(
                        val_mae[:,0],
                        dims=['Lead Time'],
                        coords={
                            'Lead Time': fc_iter.lead_time.values,
                        },
                        name='z'
                    )]
    xr_mae = xr.merge(xr_mae)
    if not os.path.exists(f'{output_dir}'):
        os.mkdir(f'{output_dir}')
    if not os.path.exists(f'{output_dir}/{da_method.lower()}'):
        os.mkdir(f'{output_dir}/{da_method.lower()}')
    # 保存分析预报循环中的背景场和预报结果
    xb_iter.to_netcdf(f'{output_dir}/{da_method.lower()}/xb_iter_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc')
    print(f'{output_dir}/{da_method.lower()}/xb_iter_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc saved!')
    fc_iter.to_netcdf(f'{output_dir}/{da_method.lower()}/fc_iter_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc')
    print(f'{output_dir}/{da_method.lower()}/fc_iter_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc saved!')
    xr_rmse.to_netcdf(f'{output_dir}/{da_method.lower()}/rmse_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc')
    print(f'{output_dir}/{da_method.lower()}/rmse_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc saved!')
    xr_acc.to_netcdf(f'{output_dir}/{da_method.lower()}/acc_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc')
    print(f'{output_dir}/{da_method.lower()}/acc_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc saved!')
    xr_mae.to_netcdf(f'{output_dir}/{da_method.lower()}/mae_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc')
    print(f'{output_dir}/{da_method.lower()}/mae_{da_method.lower()}_ensemble{ai_ensemble}_obserr{obserr_level}_obspartial{obs_partial}.nc saved!')
    np.save(f'{output_dir}/{da_method.lower()}/da_time.npy', times4da)

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--data_dir',
        type=str,
        help='path of the truth data',
        default='/tmp/dataset/data/data' #_2.8125deg
    )

    parser.add_argument(
        '--xb_dir',
        type=str,
        help='path of the background fields',
        default='/tmp/dataset/da_data/da_data/background'
    )

    parser.add_argument(
        '--obs_dir',
        type = str,
        help = 'path of the observaions',
        default='/tmp/dataset/da_data/da_data/observation'
    )

    parser.add_argument(
        '--obs_mask_dir',
        type=str,
        help = 'path of the observaional masks',
        default='/tmp/dataset/da_data/da_data/observation_mask'
    )

    parser.add_argument(
        '--pretrain_dir',
        type = str,
        help = 'path for pretrain prediction models',
        default='/tmp/dataset/pred_model'
    )
    
    parser.add_argument(
        '--damodel_dir',
        type = str,
        help = 'path for pretrain cvae assimilation models',
        default = '/tmp/pretrainmodel'
    )
    
    parser.add_argument(
        '--output_dir',
        type = str,
        help = 'path for output',
        default='/tmp/output'
    )

    parser.add_argument(
        '--init_time',
        type = int,
        help = 'init lead times of the first background fields',
        default=120
    )

    parser.add_argument(
        '--dtmodel',
        type=int,
        help = 'one time step of the prediction model',
        default=6
    )

    parser.add_argument(
        '--dt_obs',
        type=int,
        help = 'dt[h] each observations',
        default = 6
    )

    parser.add_argument(
        '--dt_da_pred',
        type=int,
        help = 'one time step of the prediction model used in 4DVar',
        default=6
    )
    
    parser.add_argument(
        '--daw',
        type=int,
        help = 'length of an assimilation window [h]',
        default=12
    )
    
    parser.add_argument(
        '--obs_partial',
        type=float,
        help = 'partial of observational grids',
        default=0.5
    )
    
    parser.add_argument(
        '--obs_single',
        type=str2bool,
        help = 'test for single observation',
        default=False
    )
    
    parser.add_argument(
        '--obserr_level',
        type=float,
        help = 'level of the observation error',
        default=0.015
    )
    
    parser.add_argument(
        '--prediction_length',
        type=int,
        help = 'length of the prediction experiment [d]',
        default = 30
    )
    
    parser.add_argument(
        '--decorrelation_time',
        type=int,
        help = 'decoorelation between each initial time [h]',
        default = 720
    )
    
    parser.add_argument(
        '--out_iter',
        type=int,
        help = 'out_iter for 4DVar',
        default = 3
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        help = 'mode of data',
        default='test'
    )
    
    parser.add_argument(
        '--da_method',
        type=str,
        help = 'method used to do assimilation',
        default='4DVar'
    )

    parser.add_argument(
        '--ai_ensemble',
        type=int,
        help='ensemble size for ai model',
        default=1
    )

    parser.add_argument(
        '--resolution',
        type=float,
        help='resolution of experiment',
        default=5.625
    )
    
    return parser

if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    xb_dir = args.xb_dir
    obs_dir = args.obs_dir
    obs_mask_dir = args.obs_mask_dir
    pretrain_dir = args.pretrain_dir
    damodel_dir = args.damodel_dir
    output_dir = args.output_dir
    init_time = args.init_time
    dtmodel = args.dtmodel
    dt_obs = args.dt_obs
    dt_da_pred = args.dt_da_pred
    daw = args.daw
    obs_partial = args.obs_partial
    obs_single = args.obs_single
    obserr_level = args.obserr_level
    prediction_length = args.prediction_length
    decorrelation_time = args.decorrelation_time
    out_iter = args.out_iter
    mode = args.mode
    da_method = args.da_method
    ai_ensemble = args.ai_ensemble
    resolution = args.resolution
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # device = torch.cuda.set_device(2) if torch.cuda.is_available() else 'cpu'
    
    pred_da_loop_inference(data_dir, xb_dir, obs_dir, obs_mask_dir, pretrain_dir, damodel_dir, output_dir, init_time, dtmodel, dt_obs, dt_da_pred, daw, obs_partial, obs_single, obserr_level, prediction_length, decorrelation_time, out_iter, mode, da_method, ai_ensemble, resolution, device)

