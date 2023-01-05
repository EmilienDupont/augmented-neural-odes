from abc import abstractmethod

import torch
from torch import nn


class ODELearnableDynamics(nn.Module):
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype, data_dim: int):
        super().__init__()
        self.device = device
        self.tensor_dtype = tensor_dtype
        # https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py#L102
        self.nfe = 0
        self.data_dim = data_dim

    def get_nfe(self):
        return self.nfe

    def set_nfe(self):
        self.nfe = 0

    @abstractmethod
    def forward(self, t, y):
        pass


class ODEFuncNN3Layer(ODELearnableDynamics):
    # copy from
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype, data_dim: int, hidden_dim: int):
        super().__init__(device, tensor_dtype, data_dim)
        self.hidden_dim = hidden_dim
        self.device = device
        self.tensor_dtype = tensor_dtype
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, self.hidden_dim, device=self.device, dtype=self.tensor_dtype),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.data_dim, device=self.device, dtype=tensor_dtype),
        )
        self.net.cuda()  # move to cuda , might be redundant as device is set in nn.Linear
        assert next(self.net.parameters()).is_cuda, "Model is not on Cuda"
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py#L105
        self.nfe += 1  # Update number of function evaluations (implicit depth)
        y_pred = self.net(y)  # can be played with !
        return y_pred
