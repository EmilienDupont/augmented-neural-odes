import datetime
import json
from typing import Type, Callable

import numpy as np
import torch
from dill.source import getsource
from torch import nn

from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchODESolver


def get_device_info():
    device_info = {}
    dummy_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_info['device_type'] = dummy_device.type
    device_info['device_count'] = torch.cuda.device_count()
    device_info['device_name'] = torch.cuda.get_device_name(dummy_device)
    return device_info


def format_timedelta(time_delta: datetime.timedelta):
    tot_seconds = time_delta.seconds
    hours = tot_seconds // 3600
    minutes = (tot_seconds // 60) % 60
    seconds = tot_seconds - hours * 3600 - minutes * 60
    return f"{hours} hours , {minutes} minutes, {seconds} seconds"


def log_train_experiment(experiment_log_filepath: str, solver: TorchODESolver, f_true_dynamic: Callable, run_type: str,
                         total_solve_calls: int, training_time_fmt: str, ode_learnable_func: Type[nn.Module], torch_config: dict,
                         data_config: dict, train_params: dict, train_loss: float, nfe: int):
    payload = dict()
    with open(experiment_log_filepath, 'w') as f:
        # metrics
        payload['train_loss'] = train_loss
        payload['total_nfe'] = str(nfe)
        payload['total_solve_calls'] = str(total_solve_calls)
        payload['avg_nfe_per_solve_call'] = str(np.round(float(nfe) / total_solve_calls, 2))
        payload['training_time_fmt'] = training_time_fmt
        # params and configs
        payload['run_type'] = run_type
        payload['solver'] = str(type(solver))
        payload['f_true_dynamics'] = str(getsource(f_true_dynamic))
        payload['ode_func'] = str(getsource(ode_learnable_func))
        payload['torch_config'] = str(torch_config)
        payload['data_config'] = str(data_config)
        payload['train_config'] = str(train_params)
        #
        json.dump(obj=payload, fp=f)
        f.flush()
        f.close()
