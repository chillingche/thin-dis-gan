import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.nn.functional as F


# class ConditionalBatchNorm2d(nn.normalization._BatchNorm):
#     def __init__(self,
#                  labels,
#                  num_features,
#                  eps=1e-5,
#                  momentum=0.1,
#                  affine=True,
#                  track_running_stats=True):
#         super().__init__(num_features, eps, momentum, affine,
#                          track_running_stats)
#         self.res

#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError("expected 2D or 3D input (got {}D input)".format(
#                 input.dim()))

#     def forward(self, input):
#         self._check_input_dim(input)
#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             self.num_batches_tracked += 1
#             if self.momentum is None:  # use cumulative moving average
#                 exponential_average_factor = 1.0 / self.num_batches_tracked.item(
#                 )
#             else:  # use exponential moving average
#                 exponential_average_factor = self.momentum
#         return F.batch_norm(input, self.running_mean, self.running_var,
#                             self.weight, self.bias, self.training
#                             or not self.track_running_stats,
#                             exponential_average_factor, self.eps)


class CSketchDiscriminator(nn.Module):
    ndf = 128

    def __init__(self, ngpu, nclass=10):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(1, self.ndf, 4, 2, 1, bias=False),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1,
                      bias=False),  # 256, 8, 8
            nn.BatchNorm2d(2 * self.ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1,
                      bias=False),  # 512, 4, 4
            nn.BatchNorm2d(4 * self.ndf),
            nn.LeakyReLU(0.2, True),
            Flat())
        self.class_branch = nn.Linear(4 * self.ndf*4*4, nclass)
        self.dis_branch = nn.Linear(4 * self.ndf*4*4, 1)

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            feature = parallel.data_parallel(self.main, input, range(self.ngpu))
            class_output = parallel.data_parallel(self.class_branch, feature, range(self.ngpu))
            dis_output = parallel.data_parallel(self.dis_branch, feature, range(self.ngpu))
        else:
            feature = self.main(input)
            class_output = self.class_branch(feature)
            dis_output = self.dis_branch(feature)
        return class_output, dis_output


class CSKetchGenerator(nn.Module):
    nz = 100
    ngf = 128
    nc = 1

    def __init__(self, ngpu, nclass=10):
        super().__init__()
        self.ngpu = ngpu
        self.nclass = nclass
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz + nclass, self.ngf * 4, 4, 1, 0,
                               bias=False),  # 512, 4, 4
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),  # 256, 8, 8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1,
                               bias=False),  # 128, 16, 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1,
                               bias=False),  # 3, 32, 32
            nn.Tanh())

    def forward(self, input, y):
        if input.is_cuda:
            y_onehot = torch.zeros([y.size(0), self.nclass, 1, 1], device=torch.device("cuda"))
        else:
            y_onehot = torch.zeros([y.size(0), self.nclass, 1, 1], device=torch.device("cpu"))
        y_onehot[torch.arange(y.size(0)), y] = 1
        input_with_y = torch.cat([y_onehot, input], 1)
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input_with_y, range(self.ngpu))
        else:
            output = self.main(input_with_y)
        return output


class CPhotoDiscriminator(nn.Module):
    ndf = 128

    def __init__(self, ngpu, nclass=10):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1,
                      bias=False),  # 256, 8, 8
            nn.BatchNorm2d(2 * self.ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1,
                      bias=False),  # 512, 4, 4
            nn.BatchNorm2d(4 * self.ndf),
            nn.LeakyReLU(0.2, True),
            Flat())
        self.class_branch = nn.Linear(4 * self.ndf*4*4, nclass)
        self.dis_branch = nn.Linear(4 * self.ndf*4*4, 1)

    def forward(self, input):
        if input.is_cuda and self.ngpu != 1:
            feature = parallel.data_parallel(self.main, input, range(self.ngpu))
            class_output = parallel.data_parallel(self.class_branch, feature, range(self.ngpu))
            dis_output = parallel.data_parallel(self.dis_branch, feature, range(self.ngpu))
        else:
            feature = self.main(input)
            class_output = self.class_branch(feature)
            dis_output = self.dis_branch(feature)
        return class_output, dis_output


class CPhotoGenerator(nn.Module):
    nz = 100
    ngf = 128
    nc = 3

    def __init__(self, ngpu, nclass=10):
        super().__init__()
        self.ngpu = ngpu
        self.nclass = nclass
        self.main = nn.Sequential(
            nn.Conv2d(self.nz + 1 + nclass, self.ngf, 4, 2, 1,
                      bias=False),  # 128, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ngf, 2 * self.ngf, 4, 2, 1,
                      bias=False),  # 256, 8, 8
            nn.BatchNorm2d(2 * self.ngf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ngf, 4 * self.ngf, 4, 2, 1,
                      bias=False),  # 512, 4, 4
            nn.BatchNorm2d(4 * self.ngf),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(
                self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),  # 256, 8, 8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1,
                               bias=False),  # 128, 16, 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1,
                               bias=False),  # 3, 32, 32
            nn.Tanh())

    def forward(self, input, z, y):
        if input.is_cuda:
            y_onehot = torch.zeros([y.size(0), self.nclass, 1, 1], device=torch.device("cuda"))
        else:
            y_onehot = torch.zeros([y.size(0), self.nclass, 1, 1], device=torch.device("cpu"))
        y_onehot[torch.arange(y.size(0)), y] = 1
        z = torch.cat([y_onehot, z], 1)
        z_img = z.expand(z.size(0), z.size(1), input.size(2), input.size(3))
        input_with_z = torch.cat([input, z_img], 1)
        if input.is_cuda and self.ngpu != 1:
            output = parallel.data_parallel(self.main, input_with_z,
                                            range(self.ngpu))
        else:
            output = self.main(input_with_z)
        return output


class Flat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        nbatch, c, h, w = input.size()
        return input.view(nbatch, c * h * w)
