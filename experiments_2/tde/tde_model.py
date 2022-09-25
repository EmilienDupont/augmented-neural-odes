import typing
from typing import List, Tuple, Iterable
import torch
import numpy.typing as npt

from experiments_2.tde.tensor_ode_solvers import TensorODESolver


class TensorDiffEq(torch.nn.Module):
    def __init__(self, input_dimensions: [int, List[int]], output_dimensions: [int, List[int]],
                 tensor_dimensions: List[int]):
        """

        Parameters
        ----------
        output_dimensions : output tensor dimension(s), can be int if input is a vector, list if input is a tensor
        input_dimensions : input tensor dimension(s), can be int if input is a vector, list if input is a tensor
        tensor_dimensions : dimensions for the internal tensor A(t)

        """
        super().__init__()
        self.output_dimension = output_dimensions
        self.input_dimension = input_dimensions
        self.tensor_dimensions = tensor_dimensions
        P_dimension = [input_dimensions] if isinstance(input_dimensions, int) else input_dimensions
        P_dimension.extend(tensor_dimensions)
        F_dimension = tensor_dimensions
        F_dimension.extend([1])  # for time-dependent F
        F_dimension.extend(output_dimensions if isinstance(output_dimensions, list) else [output_dimensions])
        self.P = torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=P_dimension)
        self.U = torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=tensor_dimensions)
        self.F = torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=F_dimension)

    def forward(self): from typing import List


import torch


class TensorDiffEq(torch.nn.Module):
    def __init__(self, input_dimensions: [int, List[int]], output_dimensions: [int, List[int]],
                 tensor_dimensions: List[int], t_span: Tuple, t_eval: List):
        """

        Parameters
        ----------
        output_dimensions : output tensor dimension(s), can be int if input is a vector, list if input is a tensor
        input_dimensions : input tensor dimension(s), can be int if input is a vector, list if input is a tensor
        tensor_dimensions : dimensions for the internal tensor A(t)

        """
        super().__init__()
        self.output_dimension = output_dimensions
        self.input_dimension = input_dimensions
        self.tensor_dimensions = tensor_dimensions
        P_dimension = tensor_dimensions
        aug_dim = [input_dimensions] if isinstance(input_dimensions, int) else input_dimensions[::-1]
        P_dimension.extend(aug_dim)
        F_dimension = tensor_dimensions
        F_dimension.extend([1])  # for time-dependent F
        F_dimension.extend(output_dimensions if isinstance(output_dimensions, list) else [output_dimensions])
        self.P = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=P_dimension))
        self.U = torch.nn.Parameter(
            torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=tensor_dimensions))
        self.F = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=F_dimension))
        # Create solver
        self.tensor_ode_solver = TensorODESolver()
        assert t_span[0] < t_span[1], "t_span[0] must be < t_span[1]"
        assert t_eval[0] >= t_span[0] and t_eval[1] <= t_span[1], "t_eval must be subset of t_span ranges"

    def forward(self, x: torch.Tensor, t_span, t_eval=None):
        assert len(x.size()) == 2, "No support for batch inputs with d>1 yet"
        z0 = torch.tensordot(a=x, b=self.P, dims=([1], [0]))
        sol = self.tensor_ode_solver.solve(y0=z0, fun=TensorDiffEq.odefunc, t_span=t_span, t_eval=t_eval)

    @staticmethod
    def odefunc(t: float, z: npt.ArrayLike, z_shape: typing.List, U: torch.Tensor, non_linear=str):
        pass
