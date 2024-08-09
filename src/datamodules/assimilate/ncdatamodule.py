import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.datamodules.assimilate.ncdataset import NCDataset

def collate_fn(batch):
    xb = torch.stack([batch[i][0] for i in range(len(batch))])
    obs = torch.stack([batch[i][1] for i in range(len(batch))])
    obs_mask = torch.stack([batch[i][2] for i in range(len(batch))])
    xt = torch.stack([batch[i][3] for i in range(len(batch))])
    return (
        xb,
        obs,
        obs_mask,
        xt
    )

class AssimDataModule(LightningDataModule):
    def __init__(
            self,
            var,
            era5_dir: str,
            background_dir: str,
            obs_dir: str,
            obs_mask_dir: str,
            init_time: int,
            obs_partial: float,
            pred_len: int,
            random_erase: bool,
            seed : int= 1024,
            batch_size: int = 64,
            num_workers: int = 0,
            shuffle: bool = True,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.train_data, self.val_data, self.test_data = None, None, None

    def _init_fn(self, worker_id):
        # 固定随机数
        np.random.seed(self.hparams.seed + worker_id)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.train_data and not self.val_data and not self.test_data:
            self.train_data = NCDataset(
                                era5_dir=self.hparams.era5_dir,
                                background_dir=self.hparams.background_dir,
                                obs_dir=self.hparams.obs_dir,
                                obs_mask_dir=self.hparams.obs_mask_dir,
                                init_time=self.hparams.init_time,
                                obs_partial=self.hparams.obs_partial,
                                mode="train",
                                var=self.hparams.var,
                                pred_len=self.hparams.pred_len,
                                random_erase=self.hparams.random_erase,
                            )
                        
            self.val_data = NCDataset(
                                era5_dir=self.hparams.era5_dir,
                                background_dir=self.hparams.background_dir,
                                obs_dir=self.hparams.obs_dir,
                                obs_mask_dir=self.hparams.obs_mask_dir,
                                init_time=self.hparams.init_time,
                                obs_partial=self.hparams.obs_partial,
                                mode="val",
                                var=self.hparams.var,
                                pred_len=self.hparams.pred_len,
                                random_erase=self.hparams.random_erase,
                            )               

            self.test_data = NCDataset(
                                era5_dir=self.hparams.era5_dir,
                                background_dir=self.hparams.background_dir,
                                obs_dir=self.hparams.obs_dir,
                                obs_mask_dir=self.hparams.obs_mask_dir,
                                init_time=self.hparams.init_time,
                                obs_partial=self.hparams.obs_partial,
                                mode="test",
                                var=self.hparams.var,
                                pred_len=self.hparams.pred_len,
                                random_erase=self.hparams.random_erase,
                            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass