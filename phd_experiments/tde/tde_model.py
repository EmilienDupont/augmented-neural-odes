import typing
from typing import List, Tuple, Iterable

import numpy as np
import scipy.integrate
import torch
import numpy.typing as npt
from scipy.integrate import solve_ivp


class TensorDiffEq(torch.nn.Module):
    def __init__(self, input_dimensions: [int, List[int]], output_dimensions: [int, List[int]],
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

        # start forward pass
        z0 = x  # redundant but useful for convention convenience
        A0 = torch.tensordot(a=z0, b=self.P, dims=([1], [0]))
        A0_flattened = torch.flatten(A0).detach().numpy()
        A_sizes = A0.size()
        sol = solve_ivp(fun=TensorDiffEq.odefunc, t_span=(0, 1), y0=A0_flattened, args=(A_sizes, self.U, None))
        A_f = torch.tensor(np.transpose(sol.y)[-1]).view(A_sizes).to(torch.float32)
        y_hat = torch.tensordot(a=A_f, b=self.F, dims=([1, 2, 3], [0, 1, 2]))
        return y_hat

    @staticmethod
    def odefunc(t: float, A_flattened: npt.ArrayLike, A_sizes: torch.Size, U: torch.Tensor, non_linear: str = None):
        """

        Parameters
        ----------
        t
        A_flattened
        A_sizes
        U
        non_linear

        Returns
        -------

        """
        # TODO
        """
        Monitor:
        1. Parameter U,P,F evolution over time
        2. Loss evolution overtime
        Experiment 
        1. Time invariant U
        2. Time-variant U as diag(t) . U
        3. Non-linearity with U
        """
        A = torch.Tensor(A_flattened).view(A_sizes)
        A_out = torch.tensordot(A, U, dims=([1, 2, 3], [0, 1, 2]))
        A_out_flattened = torch.flatten(A_out).detach().numpy()
        return A_out_flattened
