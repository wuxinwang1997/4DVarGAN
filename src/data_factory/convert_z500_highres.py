import numpy as np
import xarray as xr
from pathlib import Path
import pickle
import sys
sys.path.append('.')
from src.utils.tools import createFile
import argparse

def convert_z500(data_dir, mode='train', res=2.8125):
    fileList_train = []
    years = {'train': range(1979, 2015), 'val':range(2015, 2017), 'test':range(2017, 2019)}
    for i in years[mode]:
        fileList_train.append(f'{data_dir}/{mode}/geopotential_'+str(i)+f'_{res}deg.nc')

    means, stds, hours = [], [], []
    for i in range(len(fileList_train)):
        tmp = xr.open_mfdataset(fileList_train[i])
        # print(tmp['level'])
        tmp = tmp.sel(level=500)
        means.append(np.mean(tmp['z'].values.flatten()))
        stds.append(np.std(tmp['z'].values.flatten()))
        hours.append(len(tmp['time'].values))
        tmp.to_netcdf(f'{data_dir}/{mode}/geopotential_500hPa_'+str(years[mode][i])+f'_{res}deg.nc')

    Mean, Std = 0, 0

    for i in range(len(fileList_train)):
        Mean += means[i] * hours[i]
        Std += (stds[i]**2) * hours[i]

    Mean /= np.sum(np.asarray(hours))
    Std = np.sqrt(Std / np.sum(np.asarray(hours)))

    lon = tmp['lon'].values
    lat = tmp['lat'].values

    with open(Path(f'{data_dir}')/f'scaler.pkl', 'wb') as f:
        item = {'mean': Mean, 'std': Std, 'lon': lon, 'lat': lat}
        pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Background')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/public/home/wangwuxing01/research/weatherbench/geopotential_500_2.8125deg'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='test'
    )
    parser.add_argument(
        '--res',
        type=float,
        default=2.8125
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    mode= args.mode
    res = args.res

    convert_z500(data_dir, mode, res)