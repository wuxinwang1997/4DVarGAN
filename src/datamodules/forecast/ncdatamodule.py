import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.datamodules.forecast.ncdataset import NCDataset

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out1 = torch.stack([batch[i][1] for i in range(len(batch))])
    return (
        inp,
        out1,
    )

class ForecastDataModule(LightningDataModule):
    def __init__(
            self,
            var,
            era5_dir: str,
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
                                mode="train",
                                var=self.hparams.var
                            )
                        
            self.val_data = NCDataset(
                                era5_dir=self.hparams.era5_dir,
                                mode="val",
                                var=self.hparams.var
                            )               

            self.test_data = NCDataset(
                                era5_dir=self.hparams.era5_dir,
                                mode="test",
                                var=self.hparams.var
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