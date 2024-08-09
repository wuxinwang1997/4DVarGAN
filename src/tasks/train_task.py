from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
import os
from src import utils

log = utils.get_pylogger(__name__)

try:
    from c2net.context import prepare, upload_output

    #初始化导入数据集和预训练模型到容器内
    c2net_context = prepare()
    background_path = c2net_context.dataset_path+"/"+"background/background"
    ckpt_path = c2net_context.dataset_path+"/"+"ckpt_t850"
    era5_path = c2net_context.dataset_path+"/"+"era5/era5"
    obs_path = c2net_context.dataset_path+"/"+"obs/obs"
    obs_mask_path = c2net_context.dataset_path+"/"+"obs_mask/obs_mask"
except Exception:
    prepare = None
    print("Local environment!")

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Optional[float], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    if prepare == None:
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    else:
        if cfg.datamodule["_target_"] == "src.datamodules.assimilate.ncdatamodule.AssimDataModule":
            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, 
                                                                    era5_dir=era5_path,
                                                                    background_dir=background_path,
                                                                    obs_dir=obs_path,
                                                                    obs_mask_dir=obs_mask_path)
        else:
            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, 
                                                                    era5_dir=era5_path)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict
