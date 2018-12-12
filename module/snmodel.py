import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.parallel as parallel
from module.spectralnorm import SNConv2d, SNLinear


class Discriminator(nn.Module):
    ndf = 64

    def __init__(self, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            SNConv2d(3, self.ndf, 3, 1, 1), nn.LeakyReLU(0.1),
            SNConv2d(self.ndf, self.ndf, 4, 2, 1), nn.LeakyReLU(0.1),
            SNConv2d(self.ndf, 2 * self.ndf, 3, 1, 1), nn.LeakyReLU(0.1),
            SNConv2d(2 * self.ndf, 2 * self.ndf, 4, 2, 1), nn.LeakyReLU(0.1),
            SNConv2d(2 * self.ndf, 4 * self.ndf, 3, 1, 1), nn.LeakyReLU(0.1),
            SNConv2d(4 * self.ndf, 4 * self.ndf, 4, 2, 1), nn.LeakyReLU(0.1),
            SNConv2d(4 * self.ndf, 8 * self.ndf, 3, 1, 1), nn.LeakyReLU(0.1))
        self.dense = SNLinear(4 * 4 * 512, 1)

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            y = parallel.data_parallel(self.main, input, range(self.ngpu))
            y = y.view(-1, 4 * 4 * 512)
            y = parallel.data_parallel(self.dense, y, range(self.ngpu))
        else:
            y = self.main(input)
            y = y.view(-1, 4 * 4 * 512)
            y = self.dense(y)
        return y


class Generator(nn.Module):
    # nz = 100
    ngf = 128
    nc = 3

    def __init__(self, nz, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.linear = SNLinear(self.nz, self.ngf * 4 * 4 * 4)  # 512*4*4
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SNConv2d(self.ngf * 4, self.ngf * 2, 3, 1, 1,
                     bias=True),  # 256*8*8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            SNConv2d(self.ngf * 2, self.ngf, 3, 1, 1, bias=True),  # 128*16*16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            SNConv2d(self.ngf, 3, 3, 1, 1, bias=True),  # 3*32*32
            nn.Tanh())

    def forward(self, input):
        y = input.view(-1, self.nz)
        if input.is_cuda and self.ngpu != 1:
            y = parallel.data_parallel(self.linear, y, range(self.ngpu))
            y = y.view(-1, self.ngf * 4, 4, 4)
            y = parallel.data_parallel(self.main, y, range(self.ngpu))
        else:
            y = self.linear(y)
            y = y.view(-1, self.ngf * 4, 4, 4)
            y = self.main(y)
        return y
