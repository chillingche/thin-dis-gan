import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch.autograd import Variable


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
                    'Warning: Error occurred while copying the parameter named {}'
                    .format(name))
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
