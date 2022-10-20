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
import json
import logging
from typing import Callable, Tuple, Type
from dill.source import getsource
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

SEED = 123456789
np.random.seed(SEED)


##########################


# Data Generation methods and classes
def f_simple_linear_uncoupled_ode_1(t: float, y: np.ndarray, a: float):
    # http://web.math.ucsb.edu/~ebrahim/lin_ode_sys.pdf eqn (1)
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

    def generate(self, N, batch_size, splits):
        X = np.random.uniform(low=self.ulow, high=self.uhigh, size=(N, 2))
        tqdm.pandas(desc='Generate ODE toy data')
        Y = pd.DataFrame(data=X).progress_apply(
            lambda x: solve_ivp(fun=self.f, t_span=self.t_span, y0=x, args=self.args).y[:, -1],
            axis=1).values
        Y = np.stack(Y, axis=0)
        data_set_ = TensorDataset(torch.tensor(X, device=self.device, dtype=self.tensor_dtype),
                                  torch.tensor(Y, device=self.device, dtype=self.tensor_dtype))
        lengths = list(map(lambda x: int(np.round(x * N)), splits))
        train_set, test_set, val_set = torch.utils.data.random_split(dataset=data_set_, lengths=lengths)
        return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True), \
               DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True), \
               DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)


# Learnable derivative function (Learnable Dynamics)
class ODEFunc(nn.Module):
    # copy from
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype):
        super(ODEFunc, self).__init__()
        self.nfe = 0  # Number of function evaluations
        # https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py#L102
        self.device = device
        self.tensor_dtype = tensor_dtype
        self.net = nn.Sequential(
            nn.Linear(2, 50, device=self.device, dtype=self.tensor_dtype),
            nn.Tanh(),
            nn.Linear(50, 2, device=self.device, dtype=tensor_dtype),
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

    def get_nfe(self):
        return self.nfe

    def reset_nfe(self):
        self.nfe = 0


# Learn dynamics
def train(solver: TorchODESolver, ode_func_model: nn.Module, train_loader: DataLoader, num_epochs: int, t_span: Tuple,
          train_loss_fn: Callable, lr: float, print_freq=10):
    logger = logging.getLogger()
    optimizer = optim.Adam(ode_func_model.parameters(), lr=lr)
    loss = torch.Tensor([float('inf')])
    for epoch in tqdm(range(num_epochs), desc='epochs'):
        for batch_idx, (X, Y) in enumerate(train_loader):
            assert X.is_cuda, " X batch is not on cuda"
            assert Y.is_cuda, " Y batch is not on cuda"
            optimizer.zero_grad()
            Y_pred = solver.solve_ivp(func=ode_func_model, t_span=t_span, z0=X).zf
            loss = train_loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
        if epoch % print_freq == 0:
            logger.info(f'epoch : {epoch} loss = {loss.item()}')

    return ode_func_model, loss.item()


def evaluate(solver: TorchODESolver, ode_func_model: nn.Module, t_span: Tuple, test_set: DataLoader,
             test_loss_fn: Callable):
    batches_losses = []
    for i, (X, Y) in enumerate(test_set):
        Y_pred = solver.solve_ivp(func=ode_func_model, t_span=t_span, z0=X).zf
        loss = test_loss_fn(Y_pred, Y)
        batches_losses.append(loss.item())
    return np.mean(batches_losses)


def log_experiment_info(experiment_log_filepath: str, solver: TorchODESolver, f_true_dynamic: Callable,
                        training_time_fmt: str, ode_func: Type[nn.Module], torch_config: dict,
                        data_config: dict, train_config: dict, train_loss: float, test_loss: float, nfe: int):
    payload = dict()
    with open(experiment_log_filepath, 'w') as f:
        # results
        payload['train_loss'] = train_loss
        payload['test_loss'] = test_loss
        payload['total_nfe'] = str(nfe)
        payload['avg_nfe_per_epoch'] = str(np.round(float(nfe) / train_config['n_epochs'], 2))
        payload['training_time_fmt'] = training_time_fmt
        # params and configs
        payload['solver'] = str(type(solver))
        payload['f_true_dynamics'] = str(getsource(f_true_dynamic))
        payload['ode_func'] = str(getsource(ode_func))
        payload['torch_config'] = str(torch_config)
        payload['data_config'] = str(data_config)
        payload['train_config'] = str(train_config)
        json.dump(obj=payload, fp=f)
        f.flush()
        f.close()


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
    dry_run = True
    train_flag = True
    test_flag = True
    # Torch config

    torch_configs = {'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                     'TENSOR_DTYPE': torch.float32}
    # generate toy dataset
    data_set_config = {'t_span': (0, 1), 'f_params': {'a': 1}, 'num_batches': 1000, 'batch_size': 32,
                       'splits': [0.7, 0.2, 0.1], 'f_true_dynamics': f_simple_linear_uncoupled_ode_1}

    data_gen = ToyODEData(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'], ulow=-10,
                          uhigh=10,
                          f=f_simple_linear_uncoupled_ode_1, t_span=data_set_config['t_span'],
                          args=(data_set_config['f_params']['a'],))
    train_loader_, test_loader, val_loader = data_gen.generate(
        N=data_set_config['num_batches'] * data_set_config['batch_size'], batch_size=data_set_config['batch_size'],
        splits=data_set_config['splits'])
    # train
    train_params = {'n_epochs': 1 if dry_run else 100, 'batch_size': data_set_config['batch_size'], 'lr': 1e-3,
                    't_span': data_set_config['t_span']}
    if train_flag:
        # setup training stuff
        ode_func_model_init = ODEFunc(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'])
        epochs_print_freq = max(int(train_params['n_epochs'] / 10), 1)
        loss_fn = torch.nn.SmoothL1Loss()
        solver = TorchRK45(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'])
        # start training
        start_time = datetime.datetime.now()
        ode_func_model_fitted, train_loss = train(solver=solver, train_loader=train_loader_,
                                                  num_epochs=train_params['n_epochs'],
                                                  ode_func_model=ode_func_model_init, print_freq=epochs_print_freq,
                                                  t_span=train_params['t_span'], train_loss_fn=loss_fn,
                                                  lr=train_params['lr'])
        end_time = datetime.datetime.now()

        t_delta_fmt = format_timedelta(time_delta=end_time - start_time)

        mean_eval_loss = evaluate(solver=solver, ode_func_model=ode_func_model_fitted, t_span=train_params['t_span'],
                                  test_set=test_loader,
                                  test_loss_fn=loss_fn)
        logger.info(f'Training finished in : {t_delta_fmt}')
        logger.info(f'Train loss = {train_loss}')
        logger.info(f'Test loss = {mean_eval_loss}')
        torch.save(obj=ode_func_model_fitted.state_dict(), f="toy_neural_ode.model")
        log_experiment_info(experiment_log_filepath=f"../experiments/logs/experiment_{datetime.datetime.now()}.json",
                            solver=solver, f_true_dynamic=data_set_config['f_true_dynamics'], ode_func=ODEFunc,
                            training_time_fmt=t_delta_fmt, torch_config=torch_configs, data_config=data_set_config,
                            train_config=train_params, train_loss=train_loss, test_loss=float(mean_eval_loss),
                            nfe=ode_func_model_fitted.get_nfe())
