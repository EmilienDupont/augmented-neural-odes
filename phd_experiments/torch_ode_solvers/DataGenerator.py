import logging
from typing import Callable, Tuple

import dill
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class ToyODEDataGenerator():
    def __init__(self, device: torch.device, tensor_dtype: torch.dtype, ulow: float, uhigh: float,
                 f: Callable[[float, np.ndarray, ...], np.ndarray], t_span: Tuple, args: Tuple = None):
        self.device = device
        self.tensor_dtype = tensor_dtype
        self.ulow = ulow
        self.uhigh = uhigh
        self.f = f
        self.t_span = t_span
        self.args = args
        self.logger = logging.getLogger()

    def generate_ode(self, N, batch_size, splits, data_dim=2):
        X = np.random.uniform(low=self.ulow, high=self.uhigh, size=(N, data_dim))
        self.logger.info(f'Generating ode deta with dydt = \n{dill.source.getsource(self.f)}\n')
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

    def generate_crossing_dataset(self, N, batch_size, splits):
        pass
