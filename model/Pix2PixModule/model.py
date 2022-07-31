from torch import nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, padding_mode):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
        )

    def forward(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=32, num_residuals=4, padding_mode="zeros"):
        super().__init__()
        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            #k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Down-convolution
        self.down1 = nn.Sequential(
            #k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s1
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up1 = nn.Sequential(
            #k3n128s1 (should be k3n64s1?)
            nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.up2 = nn.Sequential(
            #k3n64s1
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k3n64s1 (should be k3n32s1?)
            nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.last = nn.Sequential(
            #k3n32s1
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k7n3s1
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode)
        )

    def forward(self, x):
        x1 = self.initial_down(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.res_blocks(x)
        x = self.up1(x)
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.up2(x + x2) 
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.last(x + x1)
        #TanH
        return torch.tanh(x)



from torch.nn.utils import spectral_norm

# PyTorch implementation by vinesmsuic
# Referenced from official tensorflow implementation: https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/network.py
# slim.convolution2d uses constant padding (zeros).
# Paper used spectral_norm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,activate=True):
        super().__init__()
        self.sn_conv = spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride, 
                padding,
                padding_mode="zeros" # Author's code used slim.convolution2d, which is using SAME padding (zero padding in pytorch) 
            ))
        self.activate = activate
        if self.activate:
            self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.sn_conv(x)
        if self.activate:
            x = self.LReLU(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        
        self.model = nn.Sequential(
            #k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),
            #k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),

            #k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),
            #k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),

            #k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),
            #k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),

            #k1n1s1
            Block(features[2], out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x)

        return x


class ExpressDetector(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128,512]):
        super().__init__()
        
        self.model = nn.Sequential(
            #k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),
            #k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),

            #k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),
            #k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),

            #k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),
            #k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[2],features[3]),
            nn.Linear(features[3],out_channels)
            
        )

    def forward(self, x):
        x = self.model(x)

        return x








