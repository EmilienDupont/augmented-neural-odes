from abc import ABC, abstractmethod
from typing import Callable, Tuple, Iterable

import torch


class TorchODESolverSolution:
    def __init__(self, zf: torch.Tensor, z_trajectory: Iterable[torch.Tensor],t_values: Iterable[float]):
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
    def __init__(self, method: str):
        self.method = method

    @abstractmethod
    def solve_ivp(self, func: Callable[[float, torch.Tensor, ...], torch.Tensor], t_span: Tuple,
                  z0: torch.Tensor) -> TorchODESolverSolution:
        pass
