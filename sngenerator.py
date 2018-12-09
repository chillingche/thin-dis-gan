import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.parallel as parallel


class SKetchGenerator(nn.Module):
    nz = 100
    ngf = 128
    nc = 1

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0,
                                   bias=False)),  # 512, 4, 4
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 4, self.ngf * 2, 4, 2, 1,
                    bias=False)),  # 256, 8, 8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 2, self.ngf, 4, 2, 1,
                    bias=False)),  # 128, 16, 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1,
                                   bias=False)),  # 3, 32, 32
            nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class PhotoGenerator(nn.Module):
    nz = 100
    ngf = 128
    nc = 3

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            spectral_norm(
                nn.Conv2d(self.nz + 1, self.ngf, 4, 2, 1,
                          bias=False)),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(self.ngf, 2 * self.ngf, 4, 2, 1,
                          bias=False)),  # 256, 8, 8
            nn.BatchNorm2d(2 * self.ngf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(2 * self.ngf, 4 * self.ngf, 4, 2, 1,
                          bias=False)),  # 512, 4, 4
            nn.BatchNorm2d(4 * self.ngf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 4, self.ngf * 2, 4, 2, 1,
                    bias=False)),  # 256, 8, 8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 2, self.ngf, 4, 2, 1,
                    bias=False)),  # 128, 16, 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            spectral_norm(
                nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1,
                                   bias=False)),  # 3, 32, 32
            nn.Tanh())

    def forward(self, input, z):
        z_img = z.expand(z.size(0), z.size(1), input.size(2), input.size(3))
        input_with_z = torch.cat([input, z_img], 1)
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input_with_z,
                                            range(self.ngpu))
        else:
            output = self.main(input_with_z)
        return output