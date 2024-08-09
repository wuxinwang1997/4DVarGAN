# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from typing import Any, Dict, Optional, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import xarray as xr
from src.utils.data_utils import NAME_TO_VAR

class NCDataset(Dataset):
    def __init__(self, 
                era5_dir,
                background_dir,
                obs_dir,
                obs_mask_dir,
                init_time,
                obs_partial,
                mode,
                var,
                pred_len,
                random_erase,
        ) -> None:
        super().__init__()
        self.mode = mode
        self.background = xr.open_mfdataset(f"{background_dir}/{mode}/*.nc", combine='by_coords')[NAME_TO_VAR[var]].sel(init_time=init_time)
        self.obs = xr.open_mfdataset(f"{obs_dir}/{mode}/*.nc", combine='by_coords')[NAME_TO_VAR[var]].sel(time=self.background["time"]).values
        self.obs_mask = xr.open_mfdataset(f"{obs_mask_dir}/{mode}/mask_paritial{obs_partial}*.nc")["mask"].sel(time=self.background["time"]).values
        self.era5 = xr.open_mfdataset(f"{era5_dir}/{mode}/*.nc", combine='by_coords')[NAME_TO_VAR[var]].sel(time=self.background["time"]).values
        self.background = self.background.values
        self.mean = np.load(os.path.join(f"{era5_dir}", 'normalize_mean.npy'))
        self.std = np.load(os.path.join(f"{era5_dir}", 'normalize_std.npy'))
        self.pred_len = pred_len
        self.random_erase = random_erase
        self.background = self.background[:-(self.pred_len+2)]
        
    def __len__(self):
        if self.random_erase:
            if self.mode == "train":
                return self.background.shape[0]
            else:
                return 4 * self.background.shape[0]
        else:
            return self.background.shape[0]

    def __getitem__(self, idx):
        #get the data
        xb = torch.from_numpy((self.background[idx % self.background.shape[0]].astype(np.float32) - self.mean) / self.std)
        obs = torch.from_numpy((self.obs[idx % self.background.shape[0]:idx % self.background.shape[0]+3].astype(np.float32) - self.mean) / self.std)
        if len(obs.shape) == 4:
            obs = np.squeeze(obs, axis=1)
        obs_mask = self.obs_mask[idx % self.background.shape[0]:idx % self.background.shape[0]+3]
        if self.random_erase:
            erase_idx = list(np.arange(0, 1.0, 0.05))
            random_idx = np.random.choice(erase_idx, 1)
            for i in range(3):
                tmp_obs_mask = np.ndarray.flatten(obs_mask[i])
                nonzero_obs_mask = np.nonzero(tmp_obs_mask)[0]
                if self.mode == 'train':
                    erase_obs_mask = np.random.choice(nonzero_obs_mask, int(random_idx * len(nonzero_obs_mask)))
                    tmp_obs_mask[erase_obs_mask] = 0
                else:
                    if idx // self.background.shape[0] == 0:
                        erase_obs_mask = np.random.choice(nonzero_obs_mask, int(0.75 * len(nonzero_obs_mask)))
                        tmp_obs_mask[erase_obs_mask] = 0
                    elif idx // self.background.shape[0] == 1:
                        erase_obs_mask = np.random.choice(nonzero_obs_mask, int(0.5 * len(nonzero_obs_mask)))
                        tmp_obs_mask[erase_obs_mask] = 0
                    elif idx // self.background.shape[0] == 2:
                        erase_obs_mask = np.random.choice(nonzero_obs_mask, int(0.25 * len(nonzero_obs_mask)))
                        tmp_obs_mask[erase_obs_mask] = 0
                obs_mask[i] = np.reshape(tmp_obs_mask, self.obs_mask[idx % self.background.shape[0]].shape)
                
        if len(obs_mask.shape) == 4:
            obs_mask = np.squeeze(obs_mask, axis=1)
        
        obs_mask = torch.from_numpy(obs_mask.astype(np.float32))
        xt = torch.from_numpy((self.era5[idx % self.background.shape[0]:idx % self.background.shape[0]+self.pred_len+1].astype(np.float32) - self.mean) / self.std)

        return xb, obs * obs_mask, obs_mask, xt
