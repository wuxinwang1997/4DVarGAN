import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class ResNet(nn.Module):

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 ):

        super().__init__()

        base_model = models.resnet18(pretrained=False)
        base_layers = list(base_model.children())
        base_layers[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, H/2, W/2)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, H/4, W/4)
        self.layer2 = base_layers[5] # size=(N, 128, H/8, W/8)
        self.layer3 = base_layers[6] # size=(N, 256, H/16, W/16)
        self.layer4 = base_layers[7] # size=(N, 512, H/32, W/32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(in_channels, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(128, out_channels=out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, input):
        # x_original = self.conv_original_size0(input)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        # x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    

class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dropout: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        super().__init__()

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dropout:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)


    def forward(self, x): return x + self.net(x)



class Generator(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 apply_dropout: bool = True,
                 num_downsampling: int = 2,
                 num_resnet_blocks: int = 4,
                 init_type: str = 'normal',
                 init_gain: float = 0.02,
                 ):

        super().__init__()

        f = 1
        num_downsampling = num_downsampling
        num_resnet_blocks = num_resnet_blocks
        self.init_type = init_type
        self.init_gain = init_gain

        conv = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(hidden_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(hidden_channels * f, hidden_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock(in_channels = hidden_channels * f, apply_dropout = apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(hidden_channels * f, hidden_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = hidden_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv]

        self.net = nn.Sequential(*self.layers)

    def init_module(self, m):
        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def forward(self, x): 
       
        x = self.net(x)

        return x

class Inrementor(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 apply_dropout: bool = True,
                 num_downsampling: int = 2,
                 num_resnet_blocks: int = 4,
                 init_type: str = 'normal',
                 init_gain: float = 0.02,
                 ):

        super().__init__()

        f = 1
        num_downsampling = num_downsampling
        num_resnet_blocks = num_resnet_blocks
        self.init_type = init_type
        self.init_gain = init_gain

        conv = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(hidden_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(hidden_channels * f, hidden_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock(in_channels = hidden_channels * f, apply_dropout = apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(hidden_channels * f, hidden_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(hidden_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = hidden_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv]

        self.net = nn.Sequential(*self.layers)

    def init_module(self, m):
        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=self.init_gain)
            else:
                raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val=0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=self.init_gain)
            nn.init.constant_(m.bias.data, val=0)

    def forward(self, x):
        xb = x[:,0:1,:,:] 
        y = x[:,1:2,:,:]
        mask = x[:,2:3,:,:]
        dy = mask * (y-xb)
        x_inc = self.net(x)
        dy_ = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.sum(dy[:,0,:,:]**2, dim=[1,2]),dim=-1),dim=-1),dim=-1)
        x = xb + x_inc * dy_/torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.count_nonzero(torch.flatten(mask[:,0,:,:], start_dim=1), dim=1),dim=-1),dim=-1),dim=-1)

        return x 