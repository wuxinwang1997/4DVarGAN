from typing import Any, Dict, Tuple
import itertools

import copy
import torch
import pickle
import numpy as np
from pytorch_lightning import LightningModule
import torchvision
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from src.utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch
from src.utils.train_utils import ImagePool

class AssimilateLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        g_A2B: torch.nn.Module,
        g_B2A: torch.nn.Module,
        d_A: torch.nn.Module,
        d_B: torch.nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        after_scheduler: torch.optim.lr_scheduler,
        mean_path: str,
        std_path: str,
        clim_paths: list,
        loss: object,
        pred_ckpt: str='',
    ) -> None:
        """Initialize a `FourCastNetLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.g_A2B = g_A2B
        weights_dict = torch.load(self.hparams.pred_ckpt)['state_dict']
        load_weights_dict = {k[4:]: v for k, v in weights_dict.items()
                             if self.g_A2B.phi_r.state_dict()[k[4:]].numel() == v.numel()}
        self.g_A2B.phi_r.load_state_dict(load_weights_dict, strict=True)
        for param in self.g_A2B.phi_r.parameters():
            param.requires_grad = False
        self.g_B2A = g_B2A
        self.d_A = d_A
        self.d_B = d_B
        
        self.ssim = SSIM()
        self.ssim_best = MaxMetric()

        # loss function
        self.criterion = self.hparams.loss
        
        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        
        decay = []
        no_decay = []
        for name, m in self.g_A2B.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)
        self.g_params = itertools.chain([*self.g_A2B.model_Grad.parameters([
            {
                "params": decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0,
            }
        ]), *self.g_B2A.parameters()])

        self.fake_pool_A = ImagePool(pool_sz=50)
        self.fake_pool_B = ImagePool(pool_sz=50)
        self.train_bias_buffer = []
        self.val_bias_buffer = []
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        
        self.mult = torch.tensor(self.std, dtype=torch.float32, requires_grad=False)
        
        self.clims = []
        for i in range(len(clim_paths)):
            clim_raw = np.load(clim_paths[i])
            clim_np = np.ones([1, 1, 32, 64])
            clim_np = ((clim_raw - self.mean) / self.std) * clim_np
            self.clims.append(torch.tensor(clim_np, dtype=torch.float32, requires_grad=False))

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse_best = MinMetric()
        self.val_acc_best = MaxMetric()

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def forward(self, xb: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor, xt: torch.Tensor, stage: str="train") -> torch.Tensor:
        """
            This is different from the training step. You should treat this as the final inference code
            (final outputs that you are looking for!), but you can definitely use it in the training_step
            to make some code reusable.
            Parameters:
                xb -- real image of A
                xt -- real image of B
        """
        with torch.set_grad_enabled(True):
            xb_ = torch.autograd.Variable(xb, requires_grad=True)
            fake_B = self.g_A2B(xb_, obs, obs_mask)
            
        fake_A = self.g_B2A(xt)
        if stage != "train":
            fake_A = fake_A.detach()
            fake_B = fake_B.detach()
        return fake_B, fake_A

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_rmse_best.reset()
        self.val_acc_best.reset()
        self.ssim_best.reset()
        self.ssim.reset()
    
    def forward_gen(self, xb, obs, obs_mask, xt, fake_A, fake_B):
        """
        Gets the remaining output of both the generators for the training/validation step
        Parameters:
            xb -- real image of xb
            xt -- real image of xt
            fake_xb -- fake image of xb
            fake_xt -- fake image of xt
        """
        cyc_A = self.g_B2A(fake_B)
        idt_A = self.g_B2A(xb)

        with torch.set_grad_enabled(True):
            fake_A_ = torch.autograd.Variable(fake_A, requires_grad=True)
            xt_ = torch.autograd.Variable(xt, requires_grad=True)
            cyc_B = self.g_A2B(fake_A_, obs, obs_mask)
            idt_B = self.g_A2B(xt_, obs, obs_mask)

        return cyc_A, idt_A, cyc_B, idt_B
    
    @staticmethod
    def forward_dis_A(dis, real_data, fake_data):

        """
        Gets the Discriminator output
        Parameters:
            dis       -- Discriminator
            real_data -- real image
            fake_data -- fake image
        """

        pred_real_data = dis(real_data)
        pred_fake_data = dis(fake_data)

        return pred_real_data, pred_fake_data
    
    @staticmethod
    def forward_dis_B(dis, real_data, fake_data, obs):

        """
        Gets the Discriminator output
        Parameters:
            dis       -- Discriminator
            real_data -- real image
            fake_data -- fake image
        """

        pred_real_data = dis(torch.concat([real_data, obs], dim=1))
        pred_fake_data = dis(torch.concat([fake_data, obs], dim=1))

        return pred_real_data, pred_fake_data
     
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        xb, obs, obs_mask, xt = batch
        obs = obs * obs_mask #+ (1-obs_mask)*xf
        fake_B, fake_A = self(xb, obs, obs_mask, xt)
            
        if optimizer_idx == 0:
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(xb, obs, obs_mask, xt, fake_A, fake_B)
                
            # Noe need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(torch.concat([fake_B, obs], dim=1))
                
            g_A2B_loss, g_B2A_loss, g_tot_loss = self.criterion.get_gen_loss(xb, xt, cyc_A, cyc_B, idt_A, idt_B,
                                                                            d_A_pred_fake_data, d_B_pred_fake_data)
                
            rec_A2B_loss = self.criterion.reconstruct_loss(fake_B, xt)
                
            dict_ = {'g_tot_train_loss': g_tot_loss,
                    'g_A2B_train_loss': g_A2B_loss,
                    'g_B2A_train_loss': g_B2A_loss,
                    'g_A2B_rec_train_loss': rec_A2B_loss,
                    }

            self.log_dict(dict_, on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True)
            torch.cuda.empty_cache()
            return g_tot_loss + rec_A2B_loss
            
        if optimizer_idx == 1:
            self.set_requires_grad([self.d_A], requires_grad=True)
            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis_A(self.d_A, xb, fake_A.detach())

            # GAN loss
            d_A_loss = self.criterion.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
                
            self.log("d_A_train_loss", d_A_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            torch.cuda.empty_cache()
            return d_A_loss
                
        if optimizer_idx == 2:
            self.set_requires_grad([self.d_B], requires_grad=True)
            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis_B(self.d_B, xt, fake_B.detach(), obs)

            # GAN loss
            d_B_loss = self.criterion.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            self.log("d_B_train_loss", d_B_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            torch.cuda.empty_cache()
            return d_B_loss
            
    def shared_step(self, batch, stage: str = 'val'):

        xb, obs, obs_mask, xt = batch
        # xf = [xb]
        # for i in range(1, obs.shape[1]):
        #     xf.append(self.g_A2B.phi_r(xf[i-1]))
        # xf = torch.concat(xf, dim=1)
        obs = obs*obs_mask #+ (1-obs_mask)*xf
        fake_B, fake_A = self(xb, obs, obs_mask, xt, stage)

        rmse = self.mult.to(self.device, dtype=fake_B.dtype) * weighted_rmse_torch(fake_B, xt)
        rmse = rmse.detach()
        acc = weighted_acc_torch(fake_B - self.clims[1].to(self.device, dtype=fake_B.dtype), 
                                        xt - self.clims[1].to(self.device, dtype=xt.dtype))
        acc = acc.detach()

        cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(xb, obs, obs_mask, xt, fake_A, fake_B)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis_A(self.d_A, xb, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis_B(self.d_B, xt, fake_B, obs)

        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss = self.criterion.get_gen_loss(xb, xt, cyc_A, cyc_B, idt_A, idt_B,
                                                                        d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.criterion.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.criterion.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        self.ssim.update(fake_B, xt.to(fake_B.dtype))

        dict_ = {f'{stage}/ssim': self.ssim,
                f'{stage}/rmse': rmse,
                f'{stage}/acc': acc,
                f'g_tot_{stage}_loss': g_tot_loss,
                f'g_A2B_{stage}_loss': g_A2B_loss,
                f'g_B2A_{stage}_loss': g_B2A_loss,
                f'd_A_{stage}_loss': d_A_loss,
                f'd_B_{stage}_loss': d_B_loss}
        
        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        return g_tot_loss, fake_B.detach(), xt.detach()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.shared_step(batch, "val")
        val_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
        val_rmse = val_rmse.detach()
        val_acc = weighted_acc_torch(preds - self.clims[1].to(self.device, dtype=preds.dtype), 
                                     targets - self.clims[1].to(self.device, dtype=preds.dtype))
        val_acc = val_acc.detach()

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", val_rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'rmse': val_rmse, 'acc': val_acc, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_rmse, val_acc = 0, 0
        for out in validation_step_outputs:
            val_rmse += out['rmse'] / len(validation_step_outputs)
            val_acc += out['acc'] / len(validation_step_outputs)

        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        self.val_rmse_best(val_rmse)  # update best so far val rmse of z500
        self.val_acc_best(val_acc)  # update best so far val acc of z500

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        self.log("val/rmse_best", self.val_rmse_best.compute(), prog_bar=True)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        ssim = self.ssim.compute()  # get current val acc
        self.ssim_best(ssim)  # update best so far val acc
        self.log("val/ssim_best", self.ssim_best.compute(), prog_bar=True)
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.inference_mode(False):
            self.g_A2B.train()
            loss, preds, targets = self.shared_step(batch, "test")
            test_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
            test_rmse = test_rmse.detach()
            test_acc = weighted_acc_torch(preds - self.clims[2].to(self.device, dtype=preds.dtype), 
                                        targets - self.clims[2].to(self.device, dtype=preds.dtype))
            test_acc = test_acc.detach()

            # update and log metrics
            self.test_loss(loss)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse", test_rmse, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        g_opt = self.hparams.g_optimizer(params=self.g_params)
        g_after_scheduler = self.hparams.after_scheduler(optimizer=g_opt)
        g_sch = self.hparams.scheduler(optimizer=g_opt, after_scheduler=g_after_scheduler)

        d_A_opt = self.hparams.d_optimizer(params=self.d_A_params)
        d_A_after_scheduler = self.hparams.after_scheduler(optimizer=d_A_opt)
        d_A_sch = self.hparams.scheduler(optimizer=d_A_opt, after_scheduler=d_A_after_scheduler)

        d_B_opt = self.hparams.d_optimizer(params=self.d_B_params)
        d_B_after_scheduler = self.hparams.after_scheduler(optimizer=d_B_opt)
        d_B_sch = self.hparams.scheduler(optimizer=d_B_opt, after_scheduler=d_B_after_scheduler)

        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]


if __name__ == "__main__":
    _ = AssimilateLitModule(None, None, None, None)