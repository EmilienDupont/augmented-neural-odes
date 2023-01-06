import logging
from typing import Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchODESolver, TorchODESolverSolution


class TorchEulerSolver(TorchODESolver):

    def __init__(self, step_size: [float, str] = 0.01):
        super().__init__(step_size)

    def solve_ivp(self, func: Callable[[float, torch.Tensor, ...], torch.Tensor], t_span: Tuple, z0: torch.Tensor,
                  args=None) -> TorchODESolverSolution:
        # step adaptation to align tf correctly
        t0, tf = t_span
        n_t = np.ceil((tf - t0) / self.step_size)
        h = (tf - t0) / n_t
        if abs(h - self.step_size) > 1e-4:
            self.logger.info(f'Modified step size from {self.step_size} to h for tf alignment')
        # start integration
        zt = z0.type(torch.float32)
        z_trajectory = [z0]
        t_values = [t0]
        if args:
            func = lambda t, y, func=func: func(t, y, *args)
        for t in tqdm(np.arange(t0, tf, h), desc='euler loop'):
            zt = zt + h * func(t, zt)
            z_trajectory.append(zt)
            t_values.append(t + h)

        return TorchODESolverSolution(zf=zt, z_trajectory=z_trajectory, t_values=t_values)
