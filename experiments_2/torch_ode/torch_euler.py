import logging
from typing import Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from experiments_2.torch_ode.torch_ode_solver import TorchODESolver, TorchODESolverSolution


class TorchEulerSolver(TorchODESolver):

    def __init__(self, method: str = "default", step_size: float = 0.01):
        super().__init__(method)
        self.step_size = step_size
        self.logger = logging.getLogger()

    def solve_ivp(self, func: Callable[[float, torch.Tensor, ...], torch.Tensor], t_span: Tuple, z0: torch.Tensor,
                  *args) -> TorchODESolverSolution:
        # step adaptation to align tf correctly
        t0, tf = t_span
        n_t = np.ceil((tf - t0) / self.step_size)
        h = (tf - t0) / n_t
        if abs(h-self.step_size)>1e-4:
            self.logger.info(f'Modified step size from {self.step_size} to h for tf alignment')
        # start integration
        zt = z0.type(torch.float32)
        z_trajectory = [z0]
        t_values = [t0]
        for t in tqdm(np.arange(t0, tf, h), desc='euler loop'):
            zt = zt + h * func(t, zt, *args)
            z_trajectory.append(zt)
            t_values.append(t + h)

        return TorchODESolverSolution(zf=zt, z_trajectory=z_trajectory, t_values=t_values)
