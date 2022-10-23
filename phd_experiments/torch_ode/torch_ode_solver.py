import logging
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Iterable

import torch


class TorchODESolverSolution:
    def __init__(self, zf: torch.Tensor, z_trajectory: Iterable[torch.Tensor], t_values: Iterable[float]):
        """
        Parameters
        ----------
        zf
        z_trajectory
        """
        self.z_trajectory = z_trajectory
        self.t_values = t_values
        self.zf = zf


class TorchODESolver(ABC):
    def __init__(self, step_size: [float, str] = 0.01):
        self.step_size = step_size
        self.logger = logging.getLogger()
        self.nfe = 0 # Number of Function Evaluations

    @abstractmethod
    def solve_ivp(self, func: Callable[[float, torch.Tensor, ...], torch.Tensor], t_span: Tuple,
                  z0: torch.Tensor,args:Tuple = None) -> TorchODESolverSolution:
        pass
