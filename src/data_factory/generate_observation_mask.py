# %% [markdown]
# # 生成背景场和观测数据
# 
# 这个notebook用于生成背景场和观测的数据
# 
# 背景场由再分析作为初始场，使用FourCastNet做逐小时预报到72小时得到。
# 
# 观测由再分析作为真值，随机采样比例5%、10%、50%、100%，再添加均值的1.5%为标准差的高斯误差得到


# %%
# Depending on your combination of package versions, this can raise a lot of TF warnings... 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
# import seaborn as sns
import pickle
import sys
sys.path.append('.')
from src.utils.score import *
from collections import OrderedDict
from src.inference.autoregressive_inference import autoregressive_inference_background
from pathlib import Path
from src.models.fourcast_module import FourCastNetLitModule
from src.utils.tools import gaussian_perturb_np
import argparse

def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')

    parser.add_argument(
        '--year',
        type=int,
        help='year of background data',
        default=1979
    )

    parser.add_argument(
        '--dt',
        type = int,
        help = 'dt of iterative prediction',
        default=24
    )

    parser.add_argument(
        '--prediction_length',
        type=int,
        help = 'prediction length of the hole iterative predcition progress',
        default=24*5
    )

    parser.add_argument(
        '--decorrelation_time',
        type = int,
        help = 'decoorelation between each objective time',
        default=6
    )

    parser.add_argument(
        '--data_dir',
        type = str,
        help = 'data dir',
        default='/dataset/z500/geopotential_500_5.625deg'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help = 'output dir of the background data',
        default='/model'
    )

    parser.add_argument(
        '--obserr_level',
        type=float,
        help = 'level of observarion error',
        default = 0.015
    )

    parser.add_argument(
        '--mode',
        type=str,
        help = 'mode of data',
        default='train'
    )

    parser.add_argument(
        '--partial',
        type=float,
        help = 'partial ratio of observations',
        default=1.0
    )

    parser.add_argument(
        '--single_obs',
        type=bool,
        help = 'single observation',
        default=False
    )
    
    return parser

def generate_observation_mask(year, dt, prediction_length, decorrelation_time, data_dir, output_dir, mode, partial, single_obs):
    # ## 读取预测数据集
    # 
    # 从.nc文件中读取数据，为后续预测技巧的验证提供基础数据支撑
    data = xr.open_mfdataset(f'{data_dir}/{mode}/geopotential_500hPa_'+str(year)+'*.nc')

    with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
        item = pickle.load(f)
        lon = item['lon']
        lat = item['lat']
        mean = item['mean']
        std = item['std']
        f.close()

    # %% [markdown]
    # ## 构建预报结果
    # 
    # 使用AFNONet做预测，将72小时预测结果写入nc文件中
    # 预报步长为1h，预报长度为7天，每隔3小时选一个初始场，每隔24小时存一次数据
    # 
    # 首先写出存下72小时预报结果的代码
    # %%
    n_samples_all = len(data['z'])
    truth_times = np.arange(prediction_length, n_samples_all, decorrelation_time)
    n_truth_times = len(truth_times)
    observations = np.zeros((len(truth_times), 1, 32, 64))
    obs_masks = np.zeros((len(truth_times), 1, 32, 64))
    # %%
    for i, truth_time in enumerate(truth_times):
        obs = data['z'].values[truth_time] 
        observations[i] = np.expand_dims(obs, axis=0)
        obs_mask = np.zeros(obs.shape[-2]*obs.shape[-1])
        if single_obs:
            obs_mask[0] = 1
        else:
            obs_mask[:int(partial*obs_mask.shape[0])] = 1
        np.random.shuffle(obs_mask)
        obs_masks[i] = np.reshape(obs_mask, (1, 32, 64))
        del obs
    
    obs_mask_nc = xr.DataArray(
            obs_masks,
            dims=['time', 'level', 'lat', 'lon'],
            coords={
                'time': data.time.values[prediction_length:n_samples_all:decorrelation_time], 
                'level': [data.level.values],
                'lat': data.lat.values, 
                'lon': data.lon.values
            },
            name='z'
        )

    print(obs_mask_nc)

    # %%
    if single_obs:
        obs_mask_nc.to_netcdf(f'{output_dir}/{mode}/mask_observations_single_{year}.nc')
    else:
        obs_mask_nc.to_netcdf(f'{output_dir}/{mode}/mask_observations_{int(partial*100)}_{year}.nc')
    del data, obs_mask_nc


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    year = args.year
    dt = args.dt
    prediction_length = args.prediction_length
    decorrelation_time = args.decorrelation_time
    data_dir = args.data_dir
    output_dir = args.output_dir
    mode = args.mode
    partial = args.partial
    single_obs = args.single_obs

    generate_observation_mask(year, dt, prediction_length, decorrelation_time, data_dir, output_dir, mode, partial, single_obs)





