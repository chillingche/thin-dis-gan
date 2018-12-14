# coding: utf-8
# revised in 2018-05-10, add something for torch compatibility
import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch.autograd import Variable
""" Backwards Compatibility
"""
__torch_version__ = 0.0
for i, x in enumerate(torch.__version__.split('.')[:-1]):
    __torch_version__ += int(x) * pow(0.1, i)


def get_torch_version():
    return __torch_version__


def weight_init(m):
    '''
    initialize conv and bn layers
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    # if isinstance(model, nn.Module):
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_normal(m.weight)
    #             if m.bias is not None:
    #                 init_constant(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init_normal(m.weight, 1.0, 0.02)
    #             if m.bias is not None:
    #                 init_constant(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init_normal(m.weight, 0, 0.01)
    #             init_constant(m.bias, 0)
