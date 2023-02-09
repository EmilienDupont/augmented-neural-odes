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
import argparse
import datetime
import logging
import os.path
from typing import Callable, Tuple

import dill.source
import torch
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from phd_experiments.torch_ode_solvers.DataGenerator import ToyODEDataGenerator
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchODESolver
from phd_experiments.torch_ode_solvers.torch_ode_utils import get_device_info, format_timedelta, log_train_experiment
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45
from phd_experiments.torch_ode_solvers.torch_train_learnable_dynamics import ODEFuncNN3Layer
from phd_experiments.torch_ode_solvers.torch_train_true_dynamics import f_ode_linear_uncoupled, f_ode_linear_coupled, \
    f_van_der_pol

SEED = 123456789
np.random.seed(SEED)

##########################


# Data Generation methods and classes

F_TRUE_DYNAMICS_MAP = {'f1': (f_ode_linear_uncoupled, 1),
                       'f2': (f_ode_linear_coupled, [[-1, 3], [4, -2]]),
                       'f3': (f_van_der_pol, 1)}


class TorchODETrainer:
    def __init__(self, num_epochs: int, torch_solver: TorchODESolver,
                 ode_func_model: nn.Module, train_params: dict, train_loader: DataLoader,
                 train_loss_fn: Callable, t_span: Tuple):
        # Training stuff
        self.train_params = train_params
        self.num_epochs = num_epochs
        self.torch_solver = torch_solver
        self.ode_func_model = ode_func_model
        self.train_loader = train_loader
        self.train_loss_fn = train_loss_fn
        self.t_span = t_span
        # Metrics
        self.train_nfe = 0
        self.train_solve_call_count = 0
        # MISC
        self.logger = logging.getLogger()
        self.is_trained = False

    def train_ode(self, verbose=False, print_freq=10):
        self.logger = logging.getLogger()
        optimizer = optim.Adam(self.ode_func_model.parameters(), lr=self.train_params['lr'])
        loss = torch.Tensor([float('inf')])
        for epoch in tqdm(range(self.num_epochs), desc='epochs'):
            for batch_idx, (X, Y) in enumerate(self.train_loader):
                assert X.is_cuda, " X batch is not on cuda"
                assert Y.is_cuda, " Y batch is not on cuda"
                optimizer.zero_grad()
                self.train_solve_call_count += 1
                Y_pred = torch_solver_.solve_ivp(func=self.ode_func_model, t_span=self.t_span, z0=X).zf
                loss = self.train_loss_fn(Y_pred, Y)
                loss.backward()
                optimizer.step()

            if verbose and epoch % print_freq == 0:
                self.logger.info(f'epoch : {epoch} loss = {loss.item()}')
        self.logger.info(f'Final train loss after {self.num_epochs} = {loss.item()}')
        # get some training statistics
        self.train_nfe = self.ode_func_model.get_nfe()
        self.is_trained = True
        return self.ode_func_model, loss.item()

    def get_nfe(self):
        return self.train_nfe

    def get_total_solve_call_count(self):
        return self.train_solve_call_count

    def reset(self):
        self.ode_func_model.reset_nfe()
        self.train_nfe = 0
        self.train_solve_call_count = 0

    def save_model(self, model_path: str):
        torch.save(obj=self.ode_func_model.state_dict(), f=model_path)


def evaluate(solver: TorchODESolver, ode_func_model: nn.Module, t_span: Tuple, test_set: DataLoader,
             test_loss_fn: Callable):
    batches_losses = []
    for i, (X, Y) in enumerate(test_set):
        Y_pred = solver.solve_ivp(func=ode_func_model, t_span=t_span, z0=X).zf
        loss = test_loss_fn(Y_pred, Y)
        batches_losses.append(loss.item())
    return np.mean(batches_losses)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', dest='dryrun', action='store_true')
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--model-prefix', type=str, required=False, default='neural_model')
    parser.add_argument('--f', type=str, choices=[f'f{i}' for i in range(1, 4)], required=True, default='f1')
    return parser


if __name__ == '__main__':
    # TODO
    """
    i) enable cuda  done
    ii) test on toy dataset and look at predictions ( are they OK ) - print y_pred vs y_actual stats
    compute train stats 1) loss convergence 2) NFE calculations, curves 3) 
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    experiment_timestamp = datetime.datetime.now().isoformat()
    # argparse
    parser_ = get_parser()
    args = parser_.parse_args()
    # get running env info w.r.t devices
    devices_info = get_device_info()
    logger.info(f'Device info: \n {devices_info}')

    torch_configs = {'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                     'TENSOR_DTYPE': torch.float32}
    # generate toy dataset
    # dataset_config = {'t_span': (0, 1), 'f_params': {'a': 1}, 'num_batches': 1000, 'batch_size': 32,
    #                   'splits': [0.7, 0.2, 0.1], 'f_true_dynamics': f_ode_linear_uncoupled}
    dataset_config = {'data_dim': 2, 't_span': (0, 1), 'f_params': F_TRUE_DYNAMICS_MAP[args.f][1], 'num_batches': 100,
                      'batch_size': 64,
                      'splits': [0.7, 0.2, 0.1], 'f_true_dynamics': F_TRUE_DYNAMICS_MAP[args.f][0]}
    data_gen = ToyODEDataGenerator(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'],
                                   ulow=-10, uhigh=10, f=dataset_config['f_true_dynamics'],
                                   t_span=dataset_config['t_span'], args=(dataset_config['f_params'],))
    train_loader_, test_loader, val_loader = data_gen.generate_ode(
        N=dataset_config['num_batches'] * dataset_config['batch_size'], batch_size=dataset_config['batch_size'],
        splits=dataset_config['splits'])
    # train

    train_params_ = {'n_epochs': 1 if args.dryrun else 2000, 'batch_size': dataset_config['batch_size'], 'lr': 1e-3,
                     't_span': dataset_config['t_span'], 'hidden_dim': 50}

    ode_func_model_init = ODEFuncNN3Layer(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'],
                                          data_dim=dataset_config['data_dim'], hidden_dim=train_params_['hidden_dim'])
    epochs_print_freq = max(int(train_params_['n_epochs'] / 10), 1)
    loss_fn = torch.nn.SmoothL1Loss()
    torch_solver_ = TorchRK45(device=torch_configs['device'], tensor_dtype=torch_configs['TENSOR_DTYPE'])
    logger.info('Starting training with \n'
                f'data configs = {dataset_config}\n'
                f'torch configs = {torch_configs}\n'
                f'train_params = {train_params_}\n')

    # start training
    start_time = datetime.datetime.now()
    torch_trainer = TorchODETrainer(num_epochs=train_params_['n_epochs'], train_loader=train_loader_,
                                    torch_solver=torch_solver_, ode_func_model=ode_func_model_init,
                                    train_params=train_params_, train_loss_fn=loss_fn,
                                    t_span=train_params_['t_span'])
    ode_func_model_fitted, train_loss = torch_trainer.train_ode(verbose=True)
    end_time = datetime.datetime.now()
    # log results
    t_delta_fmt = format_timedelta(time_delta=end_time - start_time)
    torch_trainer.save_model(
        model_path=os.path.join(args.model_dir, f'{args.model_prefix}_{experiment_timestamp}.model'))
    log_train_experiment(
        experiment_log_filepath=os.path.join(args.log_dir, f"train_experiment_{experiment_timestamp}.json"),
        run_type="train", solver=torch_solver_, f_true_dynamic=dataset_config['f_true_dynamics'],
        ode_learnable_func=ODEFuncNN3Layer, training_time_fmt=t_delta_fmt, torch_config=torch_configs,
        data_config=dataset_config, train_params=train_params_, train_loss=train_loss,
        nfe=torch_trainer.get_nfe(), total_solve_calls=torch_trainer.get_total_solve_call_count())

    # mean_eval_loss = evaluate(solver=torch_solver_, ode_func_model=ode_func_model_fitted,
    # t_span=train_hyper_params_['t_span'],
    #                           test_set=test_loader,
    #                           test_loss_fn=loss_fn)
    # logger.info(f'Training finished in : {t_delta_fmt}')
    # logger.info(f'Train loss = {train_loss}')
    # logger.info(f'Test loss = {mean_eval_loss}')
    # torch.save(obj=ode_func_model_fitted.state_dict(), f="toy_neural_ode.model")
