import torch
import torch.nn as nn

# class GANBuilder(object):
#     def __init__(self, *args, **kwargs):
#         return super().__init__(*args, **kwargs)


class GAN(object):
    """ Vanilla GAN prototype
    """

    def __init__(self):
        self.net_d = None
        self.net_g = None


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.arch = nn.Sequential()

    def forward(self, *input):
        return self.arch.forward(*input)


class Discriminator(nn.Module):
    def __init__(self):
        return super(Discriminator, self).__init__()
        self.arch = nn.Sequential()

    def forward(self, *input):
        return super().forward(*input)


class EnhanceModule(nn.Module):
    def __init__(self, arch_module):
        super(EnhanceModule, self).__init__()
        self.arch = arch_module

    def forward(self, *input):
        return self.arch(*input)

    def weight_init(self):
        pass

    def backward():
        pass


class SequentialArchBuilder(nn.Module):
    def __init__(self, prototxt):
        super(SequentialArchBuilder, self).__init__()
        self.prototxt = prototxt
        self.arch = None

    def build(self):
        if isinstance(self.prototxt, list):
            for proto in self.prototxt:
                pass

    def forward(self, *input):
        return super().forward(*input)