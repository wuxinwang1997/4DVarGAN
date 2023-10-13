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
import argparse

def prepare_parser():
    parser = argparse.ArgumentParser(description='Generate Background')

    parser.add_argument(
        '--year',
        type=int,
        help='start year of background data',
        default=1979
    )

    parser.add_argument(
        '--dtmodel',
        type=int,
        default=6
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
        default=120
    )

    parser.add_argument(
        '--decorrelation_time',
        type = int,
        help = 'decoorelation between each objective time',
        default=3
    )

    parser.add_argument(
        '--data_dir',
        type = str,
        help = 'data dir',
        default='/mnt/d/Study/Lab/work/idea3/code/data'
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help = 'output dir of the background data',
        default='/mnt/d/Study/Lab/work/idea3/code/pretrain'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help = 'output dir of the background data',
        default='/mnt/d/Study/Lab/work/idea3/code/da_data'
    )

    parser.add_argument(
        '--mode',
        type=str,
        help = 'mode of data',
        default='train'
    )

    return parser

def generate_background(year, dt, dtmodel, prediction_length, decorrelation_time, data_dir, pretrain_dir, output_dir, mode, device):
# %% [markdown]
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

    # ## 加载训练好的模型参数
    # 6小时步长
    module = FourCastNetLitModule.load_from_checkpoint(f'{pretrain_dir}/fourcastnet_5.625deg_leadtime{dtmodel}h.ckpt')
    afnonet = module.net.to(device).eval()

    # %% [markdown]
    # ## 构建预报结果
    # 
    # 使用AFNONet做预测，将120小时预测结果写入nc文件中
    # 预报步长为6h，预报长度为5天，每隔3小时选取一个初始场，每隔24小时存一次数据
    # 
    # 首先写出存下120小时预报结果的代码
    # %%
    n_samples_all = len(data['z'])
    ics = np.arange(prediction_length, n_samples_all, decorrelation_time)
    n_ics = len(ics)

    # %%
    fcs = []
    for i, ic in enumerate(ics):
        fc = autoregressive_inference_background(ic, mean, std, data, afnonet, dtmodel, dt, prediction_length, device)
        fcs.append(fc)
        del fc

    # %%
    fc_iter = xr.concat(fcs, dim='time')

    print(fc_iter)

    # %%
    fc_iter.to_netcdf(f'{output_dir}/{mode}/background_predlen{prediction_length}_dtmodel{dtmodel}_year{year}.nc')

    del data

if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    year = args.year
    dtmodel = args.dtmodel
    dt = args.dt
    prediction_length = args.prediction_length
    decorrelation_time = args.decorrelation_time
    data_dir = args.data_dir
    pretrain_dir = args.pretrain_dir
    output_dir = args.output_dir
    mode = args.mode

    device = device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    generate_background(year, dt, dtmodel, prediction_length, decorrelation_time, data_dir, pretrain_dir, output_dir, mode, device)





