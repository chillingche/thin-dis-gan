import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn


class SNConv2d(nn.Conv2d):
    """ 2D Convolutional layer with spectral normalizaition

    Args:
    Attributes:

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 n_power_iterations=1):
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but got n_power_iterations=%d"
                % n_power_iterations)
        self.n_power_iterations = n_power_iterations
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if not self.is_params_inited():
            self.init_params()

    def is_params_inited(self):
        try:
            u = getattr(self, "weight_u")
            v = getattr(self, "weight_v")
            w = getattr(self, "weight_bar")
            return True
        except AttributeError:
            return False

    def init_params(self):
        weight = self.weight
        with torch.no_grad():
            weight_mat = self.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = _l2normalize(weight.new_empty(h).normal_(0, 1))
            v = _l2normalize(weight.new_empty(w).normal_(0, 1))
        u = nn.Parameter(u, requires_grad=False)
        v = nn.Parameter(v, requires_grad=False)
        weight = nn.Parameter(weight, requires_grad=True)
        del self._parameters["weight"]
        self.register_parameter("weight_bar", weight)
        self.register_parameter("weight_u", u)
        self.register_parameter("weight_v", v)

    def reshape_weight_to_matrix(self, weight):
        """ Conv2d Weight Shape: (out_c, in_c, k, k), Bias Shape: (out_c)
        """
        weight_mat = weight
        return weight_mat.reshape(weight_mat.size(0), -1)

    def compute_weight(self, eps=1e-12, do_power_iteration=False):
        weight = getattr(self, "weight_bar")
        u = getattr(self, "weight_u")
        v = getattr(self, "weight_v")
        weight_mat = self.reshape_weight_to_matrix(weight)
        W = weight_mat
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v.data = _l2normalize(torch.mv(W.t(), u.data))
                    u.data = _l2normalize(torch.mv(W, v.data))
                if self.n_power_iterations > 0:
                    u, v = u.clone(), v.clone()
        sigma = torch.dot(u, torch.mv(W, v))
        weight = weight / (sigma + eps)
        return weight

    def forward(self, input):
        weight = self.compute_weight(do_power_iteration=self.training)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear):
    """ Linear Convolutional layer with spectral normalization
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 n_power_iterations=1):
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but got n_power_iterations=%d"
                % n_power_iterations)
        self.n_power_iterations = n_power_iterations
        super(SNLinear, self).__init__(in_features, out_features, bias=bias)
        if not self.is_params_inited():
            self.init_params()

    def is_params_inited(self):
        try:
            u = getattr(self, "weight_u")
            v = getattr(self, "weight_v")
            w = getattr(self, "weight_bar")
            return True
        except AttributeError:
            return False

    def init_params(self):
        weight = self.weight
        with torch.no_grad():
            weight_mat = self.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = _l2normalize(weight.new_empty(h).normal_(0, 1))
            v = _l2normalize(weight.new_empty(w).normal_(0, 1))
        u = nn.Parameter(u, requires_grad=False)
        v = nn.Parameter(v, requires_grad=False)
        weight = nn.Parameter(weight, requires_grad=True)
        del self._parameters["weight"]
        self.register_parameter("weight_bar", weight)
        self.register_parameter("weight_u", u)
        self.register_parameter("weight_v", v)

    def reshape_weight_to_matrix(self, weight):
        """ Linear Weight Shape: (out_c, in_c), Bias Shape: (out_c)
        """
        weight_mat = weight
        return weight_mat.reshape(weight_mat.size(0), -1)

    def compute_weight(self, eps=1e-12, do_power_iteration=False):
        weight = getattr(self, "weight_bar")
        u = getattr(self, "weight_u")
        v = getattr(self, "weight_v")
        weight_mat = self.reshape_weight_to_matrix(weight)
        W = weight_mat
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v.data = _l2normalize(torch.mv(W.t(), u.data))
                    u.data = _l2normalize(torch.mv(W, v.data))
                if self.n_power_iterations > 0:
                    u, v = u.clone(), v.clone()
        sigma = torch.dot(u, torch.mv(W, v))
        weight = weight / (sigma + eps)
        return weight

    def forward(self, input):
        weight = self.compute_weight(do_power_iteration=self.training)
        return F.linear(input, weight, self.bias)


def _l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def max_singular_value(weight_mat, device, u=None, n_power_iterations=1):
    W = weight_mat
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
    # W = torch.randn(3, 3).to(device)
    # print(max_singular_value(W, device))
    # input_t = torch.randn(1, 3, 3, 3).to(device)
    # m = SNConv2d(3, 1, 3).to(device)
    input_t = torch.randn(1, 100).to(device)
    m = SNLinear(100, 1).to(device)
    print("weight before SN: \n", m.weight_bar, "\nu: \n", m.weight_u,
          "\nv: \n", m.weight_v)
    m.train()
    output = m(input_t)
    print("========================================")
    print("weight after SN: \n", m.weight_bar, "\nu: \n", m.weight_u, "\nv: \n",
          m.weight_v)
    loss = ((output - 1)**2).mean()
    loss.backward()
    print("loss: %.4f" % loss.item())
