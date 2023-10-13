import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 2,
            hidden_channels: int = 32,
            out_channels: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=2*hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, padding='same'),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=2*hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=5, padding='same'),
        )

    def forward(self, x):
        layer1 = self.conv1(x)
        layer2 = self.conv2(layer1)
        layer3 = self.down1(layer2)
        layer4 = self.conv3(layer3)
        layer5 = self.conv4(layer4)
        layer6 = self.down2(layer5)
        layer7 = self.conv5(layer6)
        layer8 = layer6 + self.conv6(layer7)
        layer9 = self.conv7(layer8)
        layer10 = layer8 + self.conv8(layer9)
        layer11 = torch.concat([self.up1(layer10), layer5], dim=1)
        layer12 = self.conv9(layer11)
        layer13 = self.conv10(layer12)
        layer14 = torch.concat([self.up2(layer13), layer2], dim=1)
        layer15 = self.conv11(layer14)
        layer16 = self.conv12(layer15)
        x = self.conv13(layer16)

        return x
