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


# initialization in nn.init: x -> x_
xavier_uniform = None
xavier_normal = None
kaiming_normal = None
kaiming_uniform = None
init_constant = None
init_normal = None

# get the value of scalar: x[0] -> x.item()
scalarization = None

# context-manager that disabled gradient calculation
no_grad = None


class no_grad_legacy(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        print(
            "Please update your PyTorch to the latest(>=0.4.0) with 'torch.no_grad()' API."
        )
        return False


if __torch_version__ < 0.4:

    def xavier_uniform(x):
        return init.xavier_uniform(x.data)

    def xavier_normal(x):
        return init.xavier_normal(x.data)

    def kaiming_uniform(x):
        return init.kaiming_uniform(x.data)

    def kaiming_normal(x):
        return init.kaiming_normal(x.data)

    def init_constant(x, c):
        return x.data.fill_(c)

    def init_normal(x, mu, std):
        return x.data.normal_(mu, std)

    def scalarization(x):
        return x[0]

    def no_grad():
        return no_grad_legacy()

else:
    xavier_uniform = init.xavier_uniform_
    xavier_normal = init.xavier_normal_
    kaiming_normal = init.kaiming_normal_
    kaiming_uniform = init.kaiming_uniform_

    def init_constant(x, c):
        return init.constant_(x, c)

    def init_normal(x, mu, std):
        return init.normal_(x, mu, std)

    def scalarization(x):
        return x.item()

    def no_grad():
        return torch.no_grad()


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


def load_pretrained_param(net, state_dict_path, cuda=True):
    '''load partial pretrained parameters for finetune

        Arguments:
        net: instance of model inheriting nn.Module
        state_dict_path: file path ended with .pkl or .pth
        cuda: enable cuda or not
    '''
    _, ext = os.path.splitext(state_dict_path)
    self_dic = net.state_dict()
    dic = None
    if ext in {'.pkl', '.pth'}:
        if cuda:
            dic = torch.load(state_dict_path)
        else:
            dic = torch.load(
                state_dict_path, map_location=lambda storage, loc: storage)
    if dic is not None:
        for name, val in dic.items():
            if name not in self_dic:
                continue
            if isinstance(val, Parameter):
                val = val.data
            try:
                self_dic[name].copy_(val)
                # print("loading {}".format(name))
            except RuntimeError:
                print(
                    'Warning: Error occurred while copying the parameter named {}'.
                    format(name))
                continue
    return net


def stamp_model(net, attach_type, **kwargs):
    """ Name the state dict of a model
        Args:
            net: object inherit from nn.Module
            attach_type: indicate file type, E.g, "log", "model"
            kwargs: key-value pair for ``epoch`` or ``iterarion``,
                E.g, epoch=20, it=10000
    """
    net_classname = net.__class__.__name__
    ep, it = None, None
    mea = 'epoch'
    for k, v in kwargs.items():
        if k in {'epoch', 'ep', 'e'}:
            ep = v
            mea = 'epoch'
            break
        elif k in {'iteration', 'iter', 'it', 'i'}:
            it = v
            mea = 'iter'
            break
        else:
            ep = -1
    ep = ep if ep is not None else it
    if attach_type in {"model", "m"}:
        attach_type = ""
    else:
        attach_type = "{}_".format(attach_type)
    name = "{}_{}{}_{}{}".format(
        time.strftime('%h%d_%H%M', time.localtime()), attach_type,
        net_classname, mea, ep)
    return name


def save_checkpoint(net, save_folder, **kwargs):
    name = stamp_model(net, attach_type="model", **kwargs)
    save_path = os.path.join(save_folder, '{}.pth'.format(name))
    torch.save(net.state_dict(), save_path)
    print('checkpoint saved at:%s' % save_path)


def set_trainable(net_module, requires_grad=False):
    ''' finetune method invoked when some module's autograd property
     is not expecting in a finetune model
     Arguments:
     net_module: module inherit nn.Module
     requires_grad: correspond to module's requires_grad property
    '''
    for param in net_module.parameters():
        param.requires_grad = requires_grad


def lr_schedule(optimizer, init_lr, epoch, schedule=None):
    ''' for example:
    schedule = {'10':1e-3, '20':1e-4, '30':1e-5}
    '''
    lr = init_lr
    if schedule is None:
        decay_rate = 10**(epoch // 10)
        lr = init_lr * 1.0 / decay_rate
    else:
        epoch_checkpoints = list(schedule.keys())
        schedule_key_type = type(epoch_checkpoints[0])
        epoch_checkpoints = sorted(list(map(int, epoch_checkpoints)))
        for ecp in epoch_checkpoints:
            if epoch >= ecp:
                lr = schedule[schedule_key_type(ecp)]
            else:
                break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
