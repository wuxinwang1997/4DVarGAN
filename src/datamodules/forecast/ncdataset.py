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
                mode,
                var,
        ) -> None:
        super().__init__()
        self.era5_dir = era5_dir
        self.mode = mode
        print(era5_dir)
        self.data = xr.open_mfdataset(f"{era5_dir}/{mode}/*.nc", combine='by_coords')[NAME_TO_VAR[var]].values
        self.mean = np.load(os.path.join(era5_dir,'normalize_mean.npy'))
        self.std = np.load(os.path.join(era5_dir, 'normalize_std.npy'))

    def __len__(self):
        return self.data.shape[0] - 12

    def __getitem__(self, idx):
        #get the data
        xt0 = torch.from_numpy((self.data[idx].astype(np.float32) - self.mean) / self.std)
        xt1 = torch.from_numpy((self.data[idx + 6].astype(np.float32) - self.mean) / self.std)
        
        return torch.unsqueeze(xt0, dim=0), torch.unsqueeze(xt1, dim=0) 
            

