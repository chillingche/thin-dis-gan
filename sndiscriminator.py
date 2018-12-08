import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.parallel as parallel


class SketchDiscriminator(nn.Module):
    ndf = 128

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(1, self.ndf, 4, 2, 1,
                                    bias=False)),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1,
                          bias=False)),  # 256, 8, 8
            # nn.BatchNorm2d(2 * self.ndf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1,
                          bias=False)),  # 512, 4, 4
            # nn.BatchNorm2d(4 * self.ndf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(4 * self.ndf, 1, 4, 1, 0, bias=False))
            # nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class PhotoDiscriminator(nn.Module):
    ndf = 128

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(3, self.ndf, 4, 2, 1,
                                    bias=False)),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1,
                          bias=False)),  # 256, 8, 8
            # nn.BatchNorm2d(2 * self.ndf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(
                nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1,
                          bias=False)),  # 512, 4, 4
            # nn.BatchNorm2d(4 * self.ndf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(4 * self.ndf, 1, 4, 1, 0, bias=False))
            # nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)
