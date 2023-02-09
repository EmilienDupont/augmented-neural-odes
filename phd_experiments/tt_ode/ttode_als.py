from typing import Any, Callable, Tuple, Iterable, List

import torch.autograd
from torch import Tensor

from dlra.tt import TensorTrain
from phd_experiments.tn.tt import TensorTrainFixedRank
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

"""
https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html 

"""


class Forward2():
    @staticmethod
    def forward2(x: torch.Tensor, P: torch.Tensor, input_dimensions: Iterable[int],
                 W: [TensorTrainFixedRank|List[TensorTrain]], tensor_dtype: torch.dtype,
                 tt_ode_func: Callable, t_span: Tuple, basis_fn: str, basis_params: dict) -> Tuple[
        Tensor, Iterable[Tensor]]:
        z0_contract_dims = list(range(1, len(input_dimensions) + 1))
        P_contract_dims = list(range(len(input_dimensions)))
        z0 = torch.tensordot(a=x, b=P, dims=(z0_contract_dims, P_contract_dims))
        torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=tensor_dtype, is_batch=True)
        soln = torch_solver.solve_ivp(func=tt_ode_func, t_span=t_span, z0=z0,
                                      args=(W, basis_fn, basis_params))
        return soln.zf, soln.z_trajectory


class TensorTrainContainer:
    def __init__(self):
        self.tt = None


class TTOdeAls(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, P: torch.Tensor, input_dimensions: Iterable[int],
                W: [TensorTrainFixedRank | List[TensorTrain]], tt_container: TensorTrainContainer,
                tensor_dtype: torch.dtype,
                tt_ode_func: Callable, t_span: Tuple, basis_fn: str, basis_params: dict) -> torch.Tensor:
        zf, z_trajectory = Forward2.forward2(x, P, input_dimensions, W, tensor_dtype, tt_ode_func, t_span, basis_fn,
                                             basis_params)
        ctx.z_trajectory = z_trajectory
        return zf

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Any:
        return None, None, None, None, None, None, None, None, None, None, None
