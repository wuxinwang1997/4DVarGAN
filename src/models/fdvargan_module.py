from typing import Any, List

import itertools
import torch
import torch.optim as optim
import torchvision
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics import MeanSquaredError
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional import structural_similarity_index_measure
from src.utils.tools import add_sn, ImagePool

class CycleGANLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            g_A2B: torch.nn.Module,
            g_B2A: torch.nn.Module,
            d_A: torch.nn.Module,
            d_B: torch.nn.Module,
            # pred_net: torch.nn.Module,
            loss: torch.nn.Module,
            rec_loss: torch.nn.Module,
            g_optimizer: torch.optim.Optimizer,
            d_optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            after_scheduler: torch.optim.lr_scheduler,
            lambda_4dvar: int,
            ckpt: str='',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.g_A2B = g_A2B
        self.g_B2A = g_B2A
        self.d_A = d_A
        self.d_B = d_B

        weights_dict = torch.load(self.hparams.ckpt)['state_dict']
        load_weights_dict = {k[4:]: v for k, v in weights_dict.items()
                             if self.g_A2B.phi_r.state_dict()[k[4:]].numel() == v.numel()}
        self.g_A2B.phi_r.load_state_dict(load_weights_dict, strict=True)
        for p in self.g_A2B.phi_r.parameters():
            p.requires_grad = False

        self.ssim = SSIM()
        self.score = MeanMetric()
        self.ssim_best = MaxMetric()
        self.score_best = MaxMetric()

        # loss function
        if self.hparams.loss is None:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = self.hparams.loss

        if self.hparams.rec_loss is None:
            self.rec_loss = torch.nn.MSELoss()
        else:
            self.rec_loss = self.hparams.rec_loss

        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])

        self.fake_pool_A = ImagePool(pool_sz=50)
        self.fake_pool_B = ImagePool(pool_sz=50)
        self.train_bias_buffer = []
        self.val_bias_buffer = []

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

    def forward(self, real_A: torch.Tensor, real_B: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        """
            This is different from the training step. You should treat this as the final inference code
            (final outputs that you are looking for!), but you can definitely use it in the training_step
            to make some code reusable.
            Parameters:
                real_A -- real image of A
                real_B -- real image of B
        """
        with torch.set_grad_enabled(True):
            xb = torch.autograd.Variable(real_A, requires_grad=True)
            fake_B = self.g_A2B(xb, y, mask)
        fake_A = self.g_B2A(real_B)

        return fake_B, fake_A

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.score_best.reset()
        self.score.reset()
        self.ssim_best.reset()
        self.ssim.reset()

    def forward_gen(self, real_A, real_B, fake_A, fake_B, y, mask):

        """
        Gets the remaining output of both the generators for the training/validation step
        Parameters:
            real_A -- real image of A
            real_B -- real image of B
            fake_A -- fake image of A
            fake_B -- fake image of B
        """
        cyc_A = self.g_B2A(fake_B)
        with torch.set_grad_enabled(True):
            xb = torch.autograd.Variable(fake_A, requires_grad=True)
            cyc_B = self.g_A2B(xb, y, mask)

        if self.loss.lambda_idt > 0:
            idt_A = self.g_B2A(real_A)
            with torch.set_grad_enabled(True):
                xb = torch.autograd.Variable(real_B, requires_grad=True)
                idt_B = self.g_A2B(xb, y, mask)
        else:
            idt_A = cyc_A
            idt_B = cyc_B

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
    def forward_dis_B(dis, real_data, fake_data, y, mask):

        """
        Gets the Discriminator output
        Parameters:
            dis       -- Discriminator
            real_data -- real image
            fake_data -- fake image
        """

        # pred_real_data = dis(torch.concat([real_data, y, mask], dim=1))
        # pred_fake_data = dis(torch.concat([fake_data, y, mask], dim=1))
        pred_real_data = dis(torch.concat([real_data, y], dim=1))
        pred_fake_data = dis(torch.concat([fake_data, y], dim=1))

        return pred_real_data, pred_fake_data

    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, y, mask_y, real_B, clim = batch
        pred_A = self.g_A2B.phi_r(real_A)
        y = y * mask_y + (1-mask_y) * torch.concat([real_A, pred_A], dim=1)
        # y = torch.concat([y, mask_y], dim=1)
        # y = y * mask_y + (1-mask_y) * real_A
        fake_B, fake_A = self(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), y, mask_y)
        pred_fake = []
        pred_fake.append(self.g_A2B.phi_r(fake_B))
        for i in range(real_B.shape[1]-2):
            pred_fake.append(self.g_A2B.phi_r(pred_fake[-1]))

        if optimizer_idx == 0:
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_A, fake_B, y, mask_y)

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad=False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(torch.concat([fake_B, y], dim=1))

            g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), cyc_A, cyc_B, idt_A, idt_B,
                                                                        d_A_pred_fake_data, d_B_pred_fake_data)

            rec_A2B_loss = self.loss.rec_loss(torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_B)
            rec_A2B_pred_loss = 0
            for i in range(len(pred_fake)):
                rec_A2B_pred_loss += self.loss.rec_loss(torch.unsqueeze(real_B[:,i+1,:,:], dim=1), pred_fake[i])
                       
            dict_ = {'g_tot_train_loss': g_tot_loss,
                     'g_A2B_train_loss': g_A2B_loss,
                     'g_B2A_train_loss': g_B2A_loss,
                     'g_A2B_rec_train_loss': rec_A2B_loss,
                     'g_A2B_rec_pred_train_loss': rec_A2B_pred_loss,
                     }

            self.log_dict(dict_, on_step=True,
                          on_epoch=True,
                          prog_bar=True,
                          logger=True)

            # return g_tot_loss  + 20 * rec_A2B_loss + 5 * rec_A2B_pred_loss
            return g_tot_loss + self.hparams.lambda_4dvar * (rec_A2B_loss + rec_A2B_pred_loss) / (1 + len(pred_fake)) #+ self.loss.lambda_cyc * cyc_B_pred_loss / len(pred_cyc)
            # return g_tot_loss + self.hparams.lambda_4dvar * rec_A2B_pred_loss / len(pred_fake) #+ self.loss.lambda_cyc * cyc_B_pred_loss / len(pred_cyc)
            
        if optimizer_idx == 1:
            self.set_requires_grad([self.d_A], requires_grad=True)
            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis_A(self.d_A, real_A, fake_A.detach())

            # GAN loss
            d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
            rec_loss = self.rec_loss(real_A, fake_A)
            self.log("d_A_train_loss", d_A_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("rec_A_train_loss", rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return d_A_loss

        if optimizer_idx == 2:
            self.set_requires_grad([self.d_B], requires_grad=True)
            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis_B(self.d_B, torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_B.detach(), y, mask_y)

            # GAN loss
            d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            rec_loss = self.rec_loss(torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_B)

            self.log("d_B_train_loss", d_B_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("rec_B_train_loss", rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return d_B_loss

    def shared_step(self, batch, stage: str = 'val'):

        real_A, y, mask_y, real_B, clim = batch
        # y = y * mask_y# + (1-mask_y) * real_A
        # y = torch.concat([y, mask_y], dim=1)
        pred_A = self.g_A2B.phi_r(real_A)
        y = y * mask_y + (1-mask_y) * torch.concat([real_A, pred_A], dim=1)
        # y = y * mask_y + (1-mask_y) * real_A
        fake_B, fake_A = self(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), y, mask_y)
        pred_fake = []
        pred_fake.append(self.g_A2B.phi_r(fake_B))
        for i in range(real_B.shape[1] - 2):
            pred_fake.append(self.g_A2B.phi_r(pred_fake[-1]))

        rec_loss = self.rec_loss(fake_B, torch.unsqueeze(real_B[:,0,:,:], dim=1))
        rec_A2B_pred_loss = 0
        for i in range(len(pred_fake)):
            rec_A2B_pred_loss += self.rec_loss(torch.unsqueeze(real_B[:, i+1, :, :], dim=1), pred_fake[i]) / (real_B.shape[1]-1)

        cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_A, fake_B, y, mask_y)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis_A(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis_B(self.d_B, torch.unsqueeze(real_B[:,0,:,:], dim=1), fake_B, y, mask_y)

        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, torch.unsqueeze(real_B[:,0,:,:], dim=1), cyc_A, cyc_B, idt_A, idt_B,
                                                                    d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        self.ssim.update(fake_B, torch.unsqueeze(real_B[:,0,:,:], dim=1).type(fake_B.dtype))

        # self.score.update(10*(1-(rec_loss + rec_A2B_pred_loss + cyc_B_pred_loss)/(2 * len(pred_fake)+1))+structural_similarity_index_measure(fake_B, torch.unsqueeze(real_B[:,0,:,:], dim=1).type(fake_B.dtype)))
        self.score.update(10*(1-(rec_loss + rec_A2B_pred_loss)/(len(pred_fake)+1))+structural_similarity_index_measure(fake_B, torch.unsqueeze(real_B[:,0,:,:], dim=1).type(fake_B.dtype)))
        # self.score.update(10*(1-rec_A2B_pred_loss)/len(pred_fake)+structural_similarity_index_measure(fake_B, torch.unsqueeze(real_B[:,0,:,:], dim=1).type(fake_B.dtype)))

        dict_ = {f'{stage}/score': self.score,
                f'ssim_{stage}': self.ssim,
                f'rec_{stage}_loss': rec_loss,
                f'pred_{stage}_loss': rec_A2B_pred_loss,
                f'g_tot_{stage}_loss': g_tot_loss,
                f'g_A2B_{stage}_loss': g_A2B_loss,
                f'g_B2A_{stage}_loss': g_B2A_loss,
                f'd_A_{stage}_loss': d_A_loss,
                f'd_B_{stage}_loss': d_B_loss}

        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # fid = self.fid.compute()  # get current val acc
        # self.fid_best(fid)  # update best so far val acc
        ssim = self.ssim.compute()  # get current val acc
        self.ssim_best(ssim)  # update best so far val acc
        score = self.score.compute()
        self.score_best(score)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/ssim_best", self.ssim_best.compute(), prog_bar=True)
        self.log("val/score_best", self.score_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    # def lr_lambda(self, epoch):
    #
    #     fraction = (epoch - self.epoch_decay) / self.epoch_decay
    #     return 1 if epoch < self.epoch_decay else 1 - fraction
    #
    # def configure_optimizers(self):
    #
    #     # define the optimizers here
    #     g_opt = torch.optim.AdamW(self.g_params, lr=self.g_lr, betas=(self.beta_1, self.beta_2))
    #     d_A_opt = torch.optim.Adam(self.d_A_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))
    #     d_B_opt = torch.optim.Adam(self.d_B_params, lr=self.d_lr, betas=(self.beta_1, self.beta_2))
    #
    #     # define the lr_schedulers here
    #     g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=self.lr_lambda)
    #     d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=self.lr_lambda)
    #     d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=self.lr_lambda)
    #
    #     # first return value is a list of optimizers and second is a list of lr_schedulers
    #     # (you can return empty list also)
    #     return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]

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
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
