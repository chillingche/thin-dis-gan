import torch
import torch.nn as nn
import torch.nn.parallel as parallel


class Discriminator(nn.Module):
    ndf = 128

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1,
                      bias=False),  # 256, 8, 8
            nn.BatchNorm2d(2 * self.ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1,
                      bias=False),  # 512, 4, 4
            nn.BatchNorm2d(4 * self.ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * self.ndf, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class PixelShuffleGenerator(nn.Module):
    nz = 100
    ngf = 128
    nc = 3

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            PixelShuffleC2D(self.nz, self.ngf * 4, 1, 4, 0),  # 512, 4, 4
            PixelShuffleC2D(self.ngf * 4, self.ngf * 2, 3, 2, 1),  # 256, 8, 8
            PixelShuffleC2D(self.ngf * 2, self.ngf, 3, 2, 1),  # 128, 16, 16
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ngf, self.nc * 4, kernel_size=3, bias=True),
            nn.InstanceNorm2d(self.nc * 4, affine=True),
            nn.Tanh(),
            nn.PixelShuffle(2)  # 3, 32, 32
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Generator(nn.Module):
    nz = 100
    ngf = 128
    nc = 3

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf*4, 4, 1, 0, bias=False), # 512, 4, 4
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False), # 256, 8, 8
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False), # 128, 16, 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False), # 3, 32, 32
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class PixelShuffleC2D(nn.Module):
    """ Deconv2d for high resolution AE
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0):
        super().__init__()
        ps_in_planes = out_planes * stride**2
        self.ref = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            in_planes, ps_in_planes, kernel_size=kernel_size, bias=True)
        self.ins = nn.InstanceNorm2d(ps_in_planes, affine=True)
        self.relu = nn.ReLU(inplace=False)
        self.ps = nn.PixelShuffle(stride)

    def forward(self, x):
        y = self.ref(x)
        y = self.conv(y)
        if y.size(-1) != 1:
            y = self.ins(y)
        y = self.relu(y)
        y = self.ps(y)
        return y


class HingleAdvLoss(object):
    @staticmethod
    def get_d_real_loss(d_on_real_logits):
        loss = nn.functional.relu(1 - d_on_real_logits)
        return loss.mean()

    @staticmethod
    def get_d_fake_loss(d_on_fake_logits):
        loss = nn.functional.relu(1 + d_on_fake_logits)
        return loss.mean()

    @staticmethod
    def get_g_loss(d_on_g_logits):
        loss = -d_on_g_logits
        return loss.mean()