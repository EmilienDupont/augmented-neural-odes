import typing
from typing import List, Tuple, Iterable

import numpy as np
import scipy.integrate
import torch
import numpy.typing as npt
from scipy.integrate import solve_ivp

from phd_experiments.torch_ode.torch_ode_solver import TorchODESolver
from phd_experiments.torch_ode.torch_rk45 import TorchRK45


class TensorODEBLOCK(torch.nn.Module):
    def __init__(self, input_dimensions: List[int], output_dimensions: List[int],
                 tensor_dimensions: List[int], t_span: Tuple, t_eval: List = None):
        super().__init__()
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions
        self.tensor_dimensions = tensor_dimensions

        # P_dimensions
        P_sizes = [input_dimensions] if isinstance(input_dimensions, int) else input_dimensions[::-1]
        P_sizes.extend(tensor_dimensions)

        # U dimensions
        U_sizes = tensor_dimensions[::-1]
        U_sizes.extend(tensor_dimensions)
        assert len(U_sizes) % 2 == 0, "U sizes must be odd"
        # F_dimension

        F_sizes = tensor_dimensions[::-1]  # for time-dependent F
        F_sizes.extend(output_dimensions if isinstance(output_dimensions, list) else [output_dimensions])
        self.P = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=P_sizes),
                                    requires_grad=True)
        self.U = torch.nn.Parameter(
            torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=U_sizes), requires_grad=False)
        self.F = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=F_sizes),
                                    requires_grad=False)

        # Create solver
        assert t_span[0] < t_span[1], "t_span[0] must be < t_span[1]"
        if t_eval is not None:
            assert t_eval[0] >= t_span[0] and t_eval[1] <= t_span[1], "t_eval must be subset of t_span ranges"
        self.monitor = {'U': [self.U], 'P': [self.P], 'F': [self.F]}

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x
        t_span
        t_eval

        Returns
        -------

        """
        # Record parameters for monitoring

        self.monitor['U'].append(torch.clone(self.U).detach())
        self.monitor['P'].append(torch.clone(self.P).detach())
        self.monitor['F'].append(torch.clone(self.F).detach())

        # param check
        assert len(x.size()) == 2, "No support for batch inputs with d>1 yet"

        # start forward passc
        z0 = x  # redundant but useful for convention convenience
        # assume z0 sizes/dimensions are batch x d0 x d1 x ... x d_z-1
        # assume P sizes/dimensions are d_z-1 x ... x d1 x d0 x d0 x d1 x ... x (d_A-1)
        z0_contract_sizes = list(range(1, len(self.input_dimensions) + 1))
        P_contract_sizes = list(range(len(self.input_dimensions)))
        A0 = torch.tensordot(a=z0, b=self.P, dims=(z0_contract_sizes, P_contract_sizes))
        torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32, )
        sol = torch_solver.solve_ivp(func=self.ode_f, t_span=(0, 1), z0=A0, args=(self.U, None, self.tensor_dimensions))

        A_f = sol.zf
        A_contract_dims = list(range(1, len(self.tensor_dimensions) + 1))
        F_contract_dims = list(range(0, len(self.tensor_dimensions)))
        y_hat = torch.tensordot(a=A_f, b=self.F, dims=(A_contract_dims, F_contract_dims))
        return y_hat

    def ode_f(self, t: float, A: torch.Tensor, U: torch.Tensor, non_linearity: str | None,
              tensor_dimensions: List[int]):
        """

        Parameters
        ----------
        t
        A
        U
        non_linearity
        Returns
        -------

        """
        # TODO
        """
        Monitor:
        1. Parameter U,P,F evolution over timef
        2. Loss evolution overtime
        Experiment 
        1. Time invariant U
        2. Time-variant U as diag(t) . U
        3. Non-linearity with U
        """
        # FIXME : contract dims can be calculated outside derivative function to enhance the running time
        # assume A_size : batch_size X d0 x d1 x...x (d_A-1)
        A_contract_dims = list(range(1, len(tensor_dimensions) + 1))
        # assume U_size : (d_A-1)x...x d1 x d0 x d0 x d1 x....(d_A-1)
        U_contract_dims = list(range(len(tensor_dimensions)))  # Assume U_size is even

        ##
        dAdt = torch.tensordot(A, U, dims=(A_contract_dims, U_contract_dims))
        return dAdt
