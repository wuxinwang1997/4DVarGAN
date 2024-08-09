import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):

    def __init__(self,
                in_channels: int = 2,
                hidden_channels: int = 64,
                num_layers: int = 3,
                init_type: str = 'normal',
                init_gain: float = 0.02,
                ):
        """
        Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """
        super().__init__()
        in_f = 1
        out_f = 2
        self.init_type = init_type
        self.init_gain = init_gain

        conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.layers = [conv, nn.InstanceNorm2d(hidden_channels * out_f), nn.LeakyReLU(0.2, True)]

        for idx in range(1, num_layers):
            conv = nn.Conv2d(hidden_channels * in_f, hidden_channels * out_f, kernel_size=4, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * out_f), nn.LeakyReLU(0.2, inplace=True)]
            in_f = out_f
            out_f *= 2

        out_f = min(2 ** num_layers, 8)
        conv = nn.Conv2d(hidden_channels * in_f, hidden_channels * out_f, kernel_size=4, stride=1, padding=1)
        self.layers += [conv, nn.InstanceNorm2d(hidden_channels * out_f), nn.LeakyReLU(0.2, inplace=True)]

        conv = nn.Conv2d(hidden_channels * out_f, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)

        self.net.apply(self.init_module)

    def init_module(self, m):
        cls_name = m.__class__.__name__
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0)

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def forward(self, x):
        return self.net(x)