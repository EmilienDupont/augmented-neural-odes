"""
The purpose of this exercise is build a toy neural-ode from scratch using my torch-ode library
Objectives
1- get hand-on neural-ode on a toy problem with toy implementation
2- test my torch-ode rk45 implementation with toy neural-ode

NeurODE Toy Demo
https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
http://web.math.ucsb.edu/~ebrahim/lin_ode_sys.pdf

Refs
---------------
i) PyTorch + Cuda
https://cnvrg.io/pytorch-cuda/

ii) gpustat -cp tp get GPU usage%
pip install gpustat

You can query it every couple of seconds (or minutes) in the middle of the training job
https://stackoverflow.com/a/51406093

iii) Accelerate Training
https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/
"""
import datetime
import logging
from typing import Callable, Tuple

import torch
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from phd_experiments.torch_ode.torch_ode_solver import TorchODESolver
from phd_experiments.torch_ode.torch_ode_utils import get_device_info, format_timedelta
from phd_experiments.torch_ode.torch_rk45 import TorchRK45

#########################
# Global variables and settings
# FIXME make it better !
#########################
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TENSOR_DTYPE = torch.float32
SEED = 123456789
np.random.seed(SEED)


##########################


# Data Generation methods and classes
def f_true_dynamics(t: float, y: np.ndarray, a: float):
    yprime = np.empty(2)
    yprime[0] = a * y[0]
    yprime[1] = -y[1]
    return yprime


class ToyODEData():
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype, ulow: float, uhigh: float, f: Callable,
                 t_span: Tuple, args: Tuple = None):
        self.device = device
        self.tensor_dtype = tensor_dtype
        self.ulow = ulow
        self.uhigh = uhigh
        self.f = f
        self.t_span = t_span
        self.args = args

    def generate(self, N, batch_size, fractions):
        X = np.random.uniform(low=self.ulow, high=self.uhigh, size=(N, 2))
        tqdm.pandas(desc='Generate ODE toy data')
        Y = pd.DataFrame(data=X).progress_apply(
            lambda x: solve_ivp(fun=self.f, t_span=self.t_span, y0=x, args=self.args).y[:, -1],
            axis=1).values
        Y = np.stack(Y, axis=0)
        data_set_ = TensorDataset(torch.tensor(X, device=self.device, dtype=self.tensor_dtype),
                                  torch.tensor(Y, device=self.device, dtype=self.tensor_dtype))
        lengths = list(map(lambda x: int(np.round(x * N)), fractions))
        train_set, test_set, val_set = torch.utils.data.random_split(dataset=data_set_, lengths=lengths)
        return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True), \
               DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True), \
               DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)


# Learnable derivative function (Learnable Dynamics)
class ODEFunc(nn.Module):
    # copy from
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype):
        super(ODEFunc, self).__init__()
        self.device = device
        self.tensor_dtype = tensor_dtype
        self.net = nn.Sequential(
            nn.Linear(2, 50, device=self.device, dtype=self.tensor_dtype),
            nn.Tanh(),
            nn.Linear(50, 2, device=self.device, dtype=tensor_dtype),
        )
        self.net.cuda()
        assert next(self.net.parameters()).is_cuda, "Model is not on Cuda"
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y_pred = self.net(y)  # can be played with !
        return y_pred


# Learn dynamics
def train(train_loader: DataLoader, num_epochs: int, ode_func_model: nn.Module, t_span: Tuple, train_loss_fn,
          print_freq=10,
          dry_run=False):
    logger = logging.getLogger()
    num_epochs = 1 if dry_run else num_epochs
    optimizer = optim.Adam(ode_func_model.parameters(), lr=1e-3)
    torch_rk45_solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    loss = torch.Tensor([float('inf')])
    for epoch in tqdm(range(num_epochs), desc='epochs'):
        for batch_idx, (X, Y) in enumerate(train_loader):
            assert X.is_cuda, " X batch is not on cuda"
            assert Y.is_cuda, " Y batch is not on cuda"
            optimizer.zero_grad()
            Y_pred = torch_rk45_solver.solve_ivp(func=ode_func_model, t_span=t_span, z0=X).zf
            loss = train_loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
        if epoch % print_freq == 0:
            logger.info(f'epoch : {epoch} loss = {loss.item()}')

    return ode_func_model, loss


def evaluate(odefunc: nn.Module, solver: TorchODESolver, test_set: DataLoader, metric):
    pass


if __name__ == '__main__':
    # TODO
    """
    i) enable cuda 
    ii) test on toy dataset and look at predictions ( are they OK ) - print y_pred vs y_actual stats
    compute train stats 1) loss convergence 2) NFE calculations, curves 3) 
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # get running env info w.r.t devices
    devices_info = get_device_info()
    logger.info(f'Device info: \n {devices_info}')
    # set run levers
    dry_run = False
    train_flag = True
    test_flag = True

    # generate toy dataset
    t_span = 0, 1
    a = 1
    num_batches = 300
    batch_size_ = 128
    data_gen = ToyODEData(device=DEVICE, tensor_dtype=TENSOR_DTYPE, ulow=-10, uhigh=10, f=f_true_dynamics,
                          t_span=t_span, args=(a,))
    train_loader_, test_loader, val_loader = data_gen.generate(N=num_batches * batch_size_, batch_size=batch_size_,
                                                               fractions=[0.7, 0.2, 0.1])
    # train
    if train_flag:
        # setup training stuff
        ode_func_model_init = ODEFunc(device=DEVICE, tensor_dtype=TENSOR_DTYPE)

        n_epochs = 100
        epochs_print_freq = int(n_epochs / 10)
        train_loss_fn = torch.nn.SmoothL1Loss()

        # start training
        start_time = datetime.datetime.now()
        ode_func_model_trained, loss = train(train_loader=train_loader_, num_epochs=n_epochs,
                                             ode_func_model=ode_func_model_init, print_freq=epochs_print_freq,
                                             dry_run=dry_run, t_span=t_span, train_loss_fn=train_loss_fn)
        end_time = datetime.datetime.now()

        t_delta_fmt = format_timedelta(time_delta=end_time - start_time)
        logger.info(f'Training finished in : {t_delta_fmt}')
        logger.info(f'Final loss = {loss.item()}')
        torch.save(obj=ode_func_model_trained.state_dict(), f="toy_neural_ode.model")
