import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        '''
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        '''

        self.conv1 = nn.Sequential(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False), nn.InstanceNorm2d(conv_dim, affine=True), nn.LeakyReLU(0.01, inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        self.conv2 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim*2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2
        self.conv3 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim*2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2
        
        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.main_res = nn.Sequential(*layers)
        
        # Up-Sampling
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim//2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim // 2
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim//2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim // 2

        self.dconv1 = nn.Sequential(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False), nn.Tanh())

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xc = torch.cat([x, c], dim=1)

        h1 = self.conv1(xc)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.main_res(h3) + h3
        h5 = self.dconv3(h4) + h2
        h6 = self.dconv2(h5) + h1

        return self.dconv1(h6) + x


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        self.repeat_num = repeat_num

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = x
        out_feats = []
        for i in range(0, self.repeat_num):
            h = nn.Sequential(*list(self.main.children())[i*2:(i+1)*2])(h)
            if i < 4:
                out_feats.append(h.squeeze())
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze(), out_feats