import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import xarray as xr
from pathlib import Path
import pickle
import numpy as np
from src.utils.tools import gaussian_perturb_np

class Z500_DA_Dataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 xb_dir, 
                 init_time,
                 obs_dir,
                 obs_partial_mask_dir,
                 mode, 
                 partial, 
                 single_obs,
                 obserr,
                 random_erase,
                 pred_len):
        self.observations = xr.open_mfdataset(f'{obs_dir}/{mode}/observation*err{obserr}.nc', combine='by_coords')
        self.backgrounds = (xr.open_mfdataset(f'{xb_dir}/{mode}/background_pred*.nc', combine='by_coords').sel(init_time=init_time)).sel(time=self.observations['time'])
        if single_obs:
             self.obs_masks = xr.open_mfdataset(f'{obs_partial_mask_dir}/{mode}/mask_observations_single_*.nc', combine='by_coords')
        else:
            self.obs_masks = xr.open_mfdataset(f'{obs_partial_mask_dir}/{mode}/mask_observations_{int(partial*100)}_*.nc', combine='by_coords')
        self.ground_truths = xr.open_mfdataset(f'{data_dir}/{mode}/*500*.nc', combine='by_coords')
        self.ground_truths = self.ground_truths.sel(time=self.observations['time'])
        
        self.input_xb = self.backgrounds['z'].values.astype(np.float32)
        self.input_y = self.observations['z'].values.astype(np.float32)
        self.mask_y = self.obs_masks['z'].values.astype(np.float32)
        self.labels = self.ground_truths['z'].values.astype(np.float32)
        self.pred_len = pred_len
        if self.pred_len > 0:
            self.input_xb = self.input_xb[:-self.pred_len]
        self.clim = np.mean(self.labels, axis=0)
        self.random_erase = random_erase
        self.mode = mode

        # data transformations
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.mean = item['mean']
            self.std = item['std']
            f.close()

        self.transforms_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))]
        )

        self.transforms_norm_ = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.mean for i in range(self.pred_len+1)], [self.std  for i in range(self.pred_len+1)])]
        )

        self.transforms_tensor = transforms.ToTensor()

        del self.backgrounds, self.observations, self.obs_masks, self.ground_truths

    def __len__(self):
        if self.mode == 'train':
            return self.input_xb.shape[0]
        else:
            return self.input_xb.shape[0] * 4

    def __getitem__(self, idx):
        # idx_time = np.random.randint(0, 5)
        xb = self.input_xb[idx % self.input_xb.shape[0]]
        y = self.input_y[idx%self.input_xb.shape[0]]
        # random erase some observation points
        mask_y = self.mask_y[idx % self.input_xb.shape[0]]
        if self.random_erase:
            mask_y = np.ndarray.flatten(self.mask_y[idx % self.input_xb.shape[0]])
            nonzero_mask_y = np.nonzero(mask_y)[0]
            if self.mode == 'train':
                erase_mask_y = np.random.choice(nonzero_mask_y, np.random.randint(low=0, high=len(nonzero_mask_y)))
                mask_y[erase_mask_y] = 0
            else:
                if idx // self.input_xb.shape[0] == 0:
                    erase_mask_y = np.random.choice(nonzero_mask_y, int(0.75 * len(nonzero_mask_y)))
                    mask_y[erase_mask_y] = 0
                elif idx // self.input_xb.shape[0] == 1:
                    erase_mask_y = np.random.choice(nonzero_mask_y, int(0.5 * len(nonzero_mask_y)))
                    mask_y[erase_mask_y] = 0
                elif idx // self.input_xb.shape[0] == 2:
                    erase_mask_y = np.random.choice(nonzero_mask_y, int(0.25 * len(nonzero_mask_y)))
                    mask_y[erase_mask_y] = 0
            mask_y = np.reshape(mask_y, self.mask_y[idx % self.input_xb.shape[0]].shape)

        gt = self.labels[idx%self.input_xb.shape[0] : idx%self.input_xb.shape[0]+self.pred_len+1 : 1]
        # gt[1] = gaussian_perturb_np(gt[1], 0.1 * self.std)
        clim = self.clim

        if self.transforms_norm:
            xb = self.transforms_norm(xb).permute(1, 2, 0)
            y = self.transforms_norm(y).permute(1, 2, 0)
        if self.transforms_norm_:
            gt = self.transforms_norm_(np.transpose(gt, (1, 2, 0)))#.permute(1, 2, 0)
        if self.transforms_tensor:
            clim = self.transforms_tensor(self.clim)

        return xb, y, mask_y, gt, clim