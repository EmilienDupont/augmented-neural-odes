from typing import Any, Callable, Tuple, Iterable

import torch.autograd
from torch import Tensor

from phd_experiments.tn.tt import TensorTrainFixedRank
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

"""
https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html 

"""


class TensorTrainContainer:
    def __init__(self):
        self.tt = None


class TTOdeAls(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor,input_dimensions : Iterable[int], P : torch.Tensor,
                W: TensorTrainFixedRank, tt_container: TensorTrainContainer, tensor_dtype: torch.dtype,
                tt_ode_func: Callable, t_span: Tuple, basis_fn: str, basis_params: dict) -> torch.Tensor:
        z0_contract_dims = list(range(1, len(input_dimensions) + 1))
        P_contract_dims = list(range(len(input_dimensions)))
        z0 = torch.tensordot(a=x, b=self.P, dims=(z0_contract_dims, P_contract_dims))
        ctx.z0 = z0
        ctx.tt_container = tt_container
        torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=tensor_dtype, is_batch=True)
        soln = torch_solver.solve_ivp(func=tt_ode_func, t_span=t_span, z0=z0,
                                      args=(W, basis_fn, basis_params))
        z_trajectory = soln.z_trajectory
        ctx.z_trajectory = z_trajectory
        zf = soln.zf
        return zf

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Any:
        z_trajectory = ctx.z_trajectory
        x=6
        return None
        # ctx: Any, z0: torch.Tensor, W: TensorTrainFixedRank, tt_container: TensorTrainContainer,
        #         tensor_dtype: torch.dtype, tt_ode_func: Callable, t_span: Tuple, basis_fn: str,
        #         basis_params: dict
