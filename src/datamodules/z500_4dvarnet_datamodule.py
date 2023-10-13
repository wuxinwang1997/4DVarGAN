from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.datamodules.components.z500_4dvarnet_dataset import Z500_4DVarNet_Dataset
from src.data_factory.convert_z500 import convert_z500

class Z5004DVarNetDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "/dataset/era5/era5",
        xb_dir: str = "/dataset/background_dtmodel6_predlen120/background_dtmodel6_predlen120",
        init_time: int = 120,
        obs_dir: str = "/dataset/observation_err0.015/observation_err0.015",
        obs_partial_mask_dir: str = "/dataset/obs_partial_mask/obs_partial_mask",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        partial: float= 1.0,
        single_obs: bool = False,
        obserr: float = 0.015,
        random_erase: bool = False,
        pred_len: int = 0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    # def prepare_data(self):
    #     """Download data if needed.
    #     Do not use it to assign state (self.x = y).
    #     """
    #     convert_z500(self.hparams.data_dir)
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Z500_4DVarNet_Dataset(
                self.hparams.data_dir, 
                self.hparams.xb_dir, 
                self.hparams.init_time,
                self.hparams.obs_dir,
                self.hparams.obs_partial_mask_dir,
                mode='train', 
                partial=self.hparams.partial, 
                single_obs=self.hparams.single_obs,
                obserr=self.hparams.obserr,
                random_erase=self.hparams.random_erase,
            )
            self.data_val = Z500_4DVarNet_Dataset(
                self.hparams.data_dir, 
                self.hparams.xb_dir, 
                self.hparams.init_time,
                self.hparams.obs_dir,
                self.hparams.obs_partial_mask_dir,
                mode='val', 
                partial=self.hparams.partial, 
                single_obs=self.hparams.single_obs,
                obserr=self.hparams.obserr,
                random_erase=self.hparams.random_erase,
            )
            self.data_test = Z500_4DVarNet_Dataset(
                self.hparams.data_dir, 
                self.hparams.xb_dir, 
                self.hparams.init_time,
                self.hparams.obs_dir,
                self.hparams.obs_partial_mask_dir,
                mode='test', 
                partial=self.hparams.partial, 
                single_obs=self.hparams.single_obs,
                obserr=self.hparams.obserr,
                random_erase=self.hparams.random_erase,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
