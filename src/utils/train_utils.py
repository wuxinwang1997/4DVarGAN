import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class GANLoss:
  """
  This class implements different losses required to train the generators and discriminators of CycleGAN
  """

  def __init__(self, rec_loss: torch.nn.Module, loss_type: str = 'MSE', lambda_rec: float = 1.0, lambda_cyc: float = 10, lambda_idt: float = 0, lambda_adv: float = 1): #, lambda_mse: float = 1.0):
    """
    Parameters:
        loss_type: Loss Function to train CycleGAN
        lambda_:   Weightage of Cycle-consistency loss and Identity loss
    """
    self.rec_loss = rec_loss
    self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
    self.lambda_rec = lambda_rec
    self.lambda_adv = lambda_adv

  def get_dis_loss(self, dis_pred_real_data, dis_pred_fake_data):
    """
    Parameters:
        dis_pred_real_data: Discriminator's prediction on real data
        dis_pred_fake_data: Discriminator's prediction on fake data
    """

    dis_tar_real_data = torch.ones_like(dis_pred_real_data, requires_grad=False)
    dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad=False)

    loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
    loss_fake_data = self.lambda_adv * self.loss(dis_pred_fake_data, dis_tar_fake_data)

    dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5

    return dis_tot_loss

  def reconstruct_loss(self, real, fake):
      rec_loss = self.lambda_rec * self.rec_loss(real, fake)
      return rec_loss

  def get_gen_gan_loss(self, dis_pred_fake_data):
    """
    Parameters:
        dis_pred_fake_data: Discriminator's prediction on fake data
    """

    gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad=False)
    gen_tot_loss = self.lambda_adv * self.loss(dis_pred_fake_data, gen_tar_fake_data)

    return gen_tot_loss


  def get_gen_loss(self, d_B_pred_fake_data):
    """
    Implements the total Generator loss
    Sum of Cycle loss, Identity loss, and GAN loss
    """

    # GAN loss
    g_A2B_gan_loss = self.get_gen_gan_loss(d_B_pred_fake_data)

    # Total individual losses
    g_A2B_loss = g_A2B_gan_loss

    return g_A2B_loss         
            
class CycleGANLoss:
  """
  This class implements different losses required to train the generators and discriminators of CycleGAN
  """

  def __init__(self, rec_loss: torch.nn.Module, loss_type: str = 'MSE', lambda_rec: float = 1.0, lambda_cyc: float = 10, lambda_idt: float = 0, lambda_adv: float = 1): #, lambda_mse: float = 1.0):
    """
    Parameters:
        loss_type: Loss Function to train CycleGAN
        lambda_:   Weightage of Cycle-consistency loss and Identity loss
    """
    self.rec_loss = rec_loss
    self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
    self.lambda_cyc = lambda_cyc
    self.lambda_idt = lambda_idt
    self.lambda_rec = lambda_rec
    self.lambda_adv = lambda_adv

  def get_dis_loss(self, dis_pred_real_data, dis_pred_fake_data):
    """
    Parameters:
        dis_pred_real_data: Discriminator's prediction on real data
        dis_pred_fake_data: Discriminator's prediction on fake data
    """

    dis_tar_real_data = torch.ones_like(dis_pred_real_data, requires_grad=False) #- torch.from_numpy(np.random.random_sample(dis_pred_real_data.shape) * 0.01 * np.random.randint(0, 30)).to(dis_pred_real_data.device, dtype=dis_pred_real_data.dtype)
    dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad=False) #+ torch.from_numpy(np.random.random_sample(dis_pred_fake_data.shape) * 0.01 * np.random.randint(0, 30)).to(dis_pred_fake_data.device, dtype=dis_pred_fake_data.dtype)

    loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
    loss_fake_data = self.lambda_adv * self.loss(dis_pred_fake_data, dis_tar_fake_data)

    dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5

    return dis_tot_loss

  def reconstruct_loss(self, real, fake):
      rec_loss = self.lambda_rec * self.rec_loss(real, fake)
      return rec_loss

  def get_gen_gan_loss(self, dis_pred_fake_data):
    """
    Parameters:
        dis_pred_fake_data: Discriminator's prediction on fake data
    """

    gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad=False) #- torch.from_numpy(np.random.random_sample(dis_pred_fake_data.shape) * 0.01 * np.random.randint(0, 30)).to(dis_pred_fake_data.device, dtype=dis_pred_fake_data.dtype)
    gen_tot_loss = self.lambda_adv * self.loss(dis_pred_fake_data, gen_tar_fake_data)

    return gen_tot_loss

  def get_gen_cyc_loss(self, real_data, cyc_data):
    """
    Parameters:
        real_data: Real images sampled from the dataloaders
        cyc_data:  Image reconstructed after passing the real image through both the generators
                   X_recons = F * G (X_real), where F and G are the two generators
    """

    gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
    # gen_cyc_loss = torch.nn.MSELoss()(real_data, cyc_data)
    gen_tot_loss = gen_cyc_loss * self.lambda_cyc

    return gen_tot_loss

  def get_gen_idt_loss(self, real_data, idt_data):
    """
    Implements the identity loss:
        nn.L1Loss(LG_B2A(real_A), real_A)
        nn.L1Loss(LG_A2B(real_B), real_B)
    """

    gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
    gen_tot_loss = gen_idt_loss * self.lambda_idt * 0.5

    return gen_tot_loss

  def get_gen_loss(self, real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, d_A_pred_fake_data,
                   d_B_pred_fake_data):
    """
    Implements the total Generator loss
    Sum of Cycle loss, Identity loss, and GAN loss
    """

    # Cycle loss
    cyc_loss_A = self.get_gen_cyc_loss(real_A, cyc_A)
    cyc_loss_B = self.get_gen_cyc_loss(real_B, cyc_B)
    tot_cyc_loss = cyc_loss_A + cyc_loss_B

    # GAN loss
    g_A2B_gan_loss = self.get_gen_gan_loss(d_B_pred_fake_data)
    g_B2A_gan_loss = self.get_gen_gan_loss(d_A_pred_fake_data)

    # Identity loss
    if self.lambda_idt > 0:
        g_B2A_idt_loss = self.get_gen_idt_loss(real_A, idt_A)
        g_A2B_idt_loss = self.get_gen_idt_loss(real_B, idt_B)
    else:
        g_B2A_idt_loss = g_A2B_idt_loss = 0

    # Total individual losses
    g_A2B_loss = g_A2B_gan_loss + tot_cyc_loss + g_A2B_idt_loss
    g_B2A_loss = g_B2A_gan_loss + tot_cyc_loss + g_B2A_idt_loss
    g_tot_loss = g_A2B_loss + g_B2A_loss - tot_cyc_loss

    return g_A2B_loss, g_B2A_loss, g_tot_loss


class ImagePool:

  """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
  """

  def __init__(self, pool_sz: int = 50):

      """
      Parameters:
          pool_sz: Size of the image buffer
      """

      self.nb_images = 0
      self.image_pool = []
      self.pool_sz = pool_sz

  def push_and_pop(self, images):

      """
      Parameters:
          images: latest images generated by the generator
      Returns a batch of images from pool!
      """

      images_to_return = []
      for image in images:
        image = torch.unsqueeze(image, 0)

        if self.nb_images < self.pool_sz:
          self.image_pool.append(image)
          images_to_return.append(image)
          self.nb_images += 1
        else:
          if np.random.uniform(0, 1) > 0.5:

            rand_int = np.random.randint(0, self.pool_sz)
            temp_img = self.image_pool[rand_int].clone()
            self.image_pool[rand_int] = image
            images_to_return.append(temp_img)
          else:
            images_to_return.append(image)

      return torch.cat(images_to_return, 0)