import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from pathlib import Path
import pickle

def createFile(filePath):
    if os.path.exists(filePath):
        print('%s:存在'%filePath)
    else:
        try:
            os.mkdir(filePath)
            print('新建文件夹：%s'%filePath)
        except Exception as e:
            os.makedirs(filePath)
            print('新建多层文件夹：%s' % filePath)

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

class WeightedMSELoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self, data_dir):
        super(WeightedMSELoss, self).__init__()
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.lat = torch.from_numpy(item['lat'])
            f.close()
        weights_lat = torch.cos(torch.deg2rad(self.lat))
        weights_lat /= torch.mean(weights_lat)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        self.weights_lat = torch.unsqueeze(weights_lat, dim=-1)
    def forward(self, output, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        error = output - target
        mse = torch.mean(((error)**2 * self.weights_lat.to(error.device)))
        return mse

class WeightedACCLoss(torch.nn.Module):
    def __init__(self, data_dir):
        super(WeightedACCLoss, self).__init__()
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.lat = torch.from_numpy(item['lat'])
            self.mean = item["mean"]
            self.std = item["std"]
            f.close()
        weights_lat = torch.cos(torch.deg2rad(self.lat))
        weights_lat /= torch.mean(weights_lat)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        self.weights_lat = torch.unsqueeze(weights_lat, dim=-1)
    
    def forward(self, da_fc, da_true, clim):
        fa = da_fc*self.std+self.mean - clim
        a = da_true*self.std+self.mean - clim
        fa_prime = fa - torch.mean(fa)
        a_prime = a - torch.mean(a)

        acc = (
                torch.sum(self.weights_lat.to(da_fc.device) * fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(self.weights_lat.to(da_fc.device) * fa_prime ** 2) * torch.sum(self.weights_lat.to(da_fc.device) * a_prime ** 2)
                )
        )
        return 1 - acc

class Weighted_MSE_ACC_Loss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self, data_dir):
        super(Weighted_MSE_ACC_Loss, self).__init__()
        self.w_mse = WeightedMSELoss(data_dir)
        self.w_acc = WeightedACCLoss(data_dir)
    def forward(self, output, target, clim):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        mse = self.w_mse(output, target)
        acc = self.w_acc(output, target, clim)
        return mse + acc

class WeightedMAELoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self, data_dir):
        super(WeightedMAELoss, self).__init__()
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.lat = torch.from_numpy(item['lat'])
            f.close()
        weights_lat = torch.cos(torch.deg2rad(self.lat))
        weights_lat /= torch.mean(weights_lat)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        weights_lat = torch.unsqueeze(weights_lat, dim=0)
        self.weights_lat = (1 + torch.unsqueeze(weights_lat, dim=-1)) / 2
    def forward(self, output, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        error = output - target
        mae = torch.mean((torch.abs(error) * self.weights_lat.to(error.device)))
        return mae

class Weighted_MSE_MAE_ACC_Loss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(Weighted_MSE_MAE_ACC_Loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss() # WeightedMSELoss()
        self.mae_loss = torch.nn.L1Loss() # WeightedMAELoss()
    
    def acc(self, da_fc, da_true, clim):
        fa = da_fc*self.std+self.mean - clim
        a = da_true*self.std+self.mean - clim
        # w = self.weights_lat.to(da_fc.device, dtype=torch.float)
        fa_prime = fa - torch.mean(fa)
        a_prime = a - torch.mean(a)

        acc = (
                torch.sum(fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(fa_prime ** 2) * torch.sum(a_prime ** 2)
                )
        )
        return 1 - acc
    
    def forward(self, output, target, clim):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        mse = self.mse_loss(output, target)
        mae = self.mae_loss(output, target)
        acc_loss = self.acc(output, target, clim)
        return mse + mae + acc_loss

class ACCLoss(torch.nn.Module):
    def __init__(self, data_dir):
        super(ACCLoss, self).__init__()
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.lat = torch.from_numpy(item['lat'])
            self.mean = item["mean"]
            self.std = item["std"]
            f.close()
    
    def forward(self, da_fc, da_true, clim):
        fa = da_fc*self.std+self.mean - clim
        a = da_true*self.std+self.mean - clim
        fa_prime = fa - torch.mean(fa)
        a_prime = a - torch.mean(a)

        acc = (
                torch.sum(fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(fa_prime ** 2) * torch.sum(a_prime ** 2)
                )
        )
        return 1 - acc
    
def gaussian_perturb(x, level):
    noise = level * torch.randn(x.shape).to(x.device, dtype=x.dtype)
    return x + noise

def gaussian_perturb_np(x, level, mean):
    h, w = x.shape
    noise = level * np.random.randn(h, w) * mean
    return x + noise

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            return nn.utils.parametrizations.spectral_norm(m)
        else:
            return m
    return m

def compute_gradient_penalty_div(real_out, fake_out, real_images, fake_images, k=2, p=6):
    '''
    compute the gradient with the paper wgan-div

    :param real_out: D output from the real images
    :param fake_out: D output from the fake images
    :param real_images: real images
    :param fake_images: fake images
    :param k: Defaults to 2.
    :param p: Defaults to 6.
    :return: computed the gradient penalty
    '''
    real_grad = torch.autograd.grad(
        outputs=real_out,
        inputs=real_images,
        grad_outputs=torch.ones(real_images.shape[0], 1, 1, 1).to(real_out.device, dtype=torch.float),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]

    real_grad_norm = real_grad.view(real_grad.shape[0], -1).pow(2).sum(1) ** (p / 2)

    fake_grad = torch.autograd.grad(
        outputs=fake_out,
        inputs=fake_images,
        grad_outputs=torch.ones(fake_images.shape[0], 1, 1, 1).to(fake_out.device, dtype=torch.float),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]

    fake_grad_norm = fake_grad.view(fake_grad.shape[0], -1).pow(2).sum(1) ** (p / 2)

    gradient_penalty = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    return gradient_penalty

def wasserstein_distance(real_scores, fake_scores):
    return torch.mean(real_scores) - torch.mean(fake_scores)


class CycleGANLoss:
  """
  This class implements different losses required to train the generators and discriminators of CycleGAN
  """

  def __init__(self, rec_loss: torch.nn.Module, loss_type: str = 'MSE', lambda_mse: float = 1.0, lambda_cyc: float = 10, lambda_idt: float = 0, lambda_adv: float = 1): #, lambda_mse: float = 1.0):
    """
    Parameters:
        loss_type: Loss Function to train CycleGAN
        lambda_:   Weightage of Cycle-consistency loss and Identity loss
    """
    self.rec_loss = rec_loss
    self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
    self.lambda_cyc = lambda_cyc
    self.lambda_idt = lambda_idt
    self.lambda_mse = lambda_mse
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
      rec_loss = self.lambda_mse * self.rec_loss(real, fake)
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

class WeightedCycleGANLoss:
  """
  This class implements different losses required to train the generators and discriminators of CycleGAN
  """

  def __init__(self, data_dir: str, rec_loss: torch.nn.Module, loss_type: str = 'MSE', lambda_mse: float = 1.0, lambda_cyc: float = 10, lambda_idt: float = 0, lambda_adv: float = 1): #, lambda_mse: float = 1.0):
    """
    Parameters:
        loss_type: Loss Function to train CycleGAN
        lambda_:   Weightage of Cycle-consistency loss and Identity loss
    """
    self.rec_loss = rec_loss
    self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
    self.lambda_cyc = lambda_cyc
    self.lambda_idt = lambda_idt
    self.lambda_mse = lambda_mse
    self.lambda_adv = lambda_adv
    self.l1loss = WeightedMAELoss(data_dir)

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
      rec_loss = self.lambda_mse * self.rec_loss(real, fake)
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

    gen_cyc_loss = self.l1loss(real_data, cyc_data)
    # gen_cyc_loss = torch.nn.MSELoss()(real_data, cyc_data)
    gen_tot_loss = gen_cyc_loss * self.lambda_cyc

    return gen_tot_loss

  def get_gen_idt_loss(self, real_data, idt_data):
    """
    Implements the identity loss:
        nn.L1Loss(LG_B2A(real_A), real_A)
        nn.L1Loss(LG_A2B(real_B), real_B)
    """

    gen_idt_loss = self.l1loss(real_data, idt_data)
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


class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, mode, normalize_shape, adanorm_scale=1, sigma=0.5, mean_detach=False, std_detach=False, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = mode
        self.sigma = sigma
        self.normalize_shape = normalize_shape
        self.adanorm_scale = adanorm_scale
        self.mean_detach = mean_detach
        self.std_detach = std_detach
        if self.mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(normalize_shape))
            self.bias = nn.Parameter(torch.Tensor(normalize_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == 'no_norm':
            return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.normalize_shape * self.sigma), 1)
            input = input.view(1, -1, self.normalize_shape)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.normalize_shape)
            return F.layer_norm(
                input, torch.Size([self.normalize_shape]), self.weight, self.bias, self.eps)
        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm * self.adanorm_scale
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm
        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output
