import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn


class SNConv2d(nn.Conv2d):
    """ 2D Convolutional layer with spectral normalizaition

    Args:
    Attributes:

    """
    pass


class SNLinear(nn.Linear):
    """ Linear Convolutional layer with spectral normalization
    """
    pass


def _l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def max_singular_value(W, device, u=None, n_power_iterations=1):
    if u is None:
        u = torch.randn(W.size(0), device=device)
    _u = u
    _v = None
    with torch.no_grad():
        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            _v = _l2normalize(torch.mv(W.t(), _u))
            _u = _l2normalize(torch.mv(W, _v))
    sigma = torch.dot(_u, torch.mv(W, _v))
    return sigma, _u, _v


if __name__ == "__main__":
    device = torch.device("cuda")
    W = torch.randn(3, 3).to(device)
    print(max_singular_value(W, device))
