import numpy as np
import xarray as xr
from pathlib import Path
import pickle
import sys
sys.path.append('/code')
from src.utils.tools import createFile

def convert_z500(data_dir, mode='train'):
    fileList_train = []
    fileList_test = []
    for i in range(1979, 2016):
        fileList_train.append(f'{data_dir}/{mode}/geopotential_500hPa_'+str(i)+'_5.625deg.nc')

    means, stds, hours = [], [], []
    for file_name in fileList_train:
        tmp = xr.open_mfdataset(file_name)
        means.append(np.mean(tmp['z'].values.flatten()))
        stds.append(np.std(tmp['z'].values.flatten()))
        hours.append(len(tmp['time'].values))

    Mean, Std = 0, 0

    for i in range(len(fileList_train)):
        Mean += means[i] * hours[i]
        Std += (stds[i]**2) * hours[i]

    Mean /= np.sum(np.asarray(hours))
    Std = np.sqrt(Std / np.sum(np.asarray(hours)))

    lon = tmp['lon'].values
    lat = tmp['lat'].values


    createFile('/model/data')

    with open(Path('/model/data')/f'scaler.pkl', 'wb') as f:
        item = {'mean': Mean, 'std': Std, 'lon': lon, 'lat': lat}
        pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()



