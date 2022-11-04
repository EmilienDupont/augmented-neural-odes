from typing import List, Tuple

import numpy as np
import torch
from torch import vmap
# https://pytorch.org/tutorials/prototype/vmap_recipe.html
from phd_experiments.tde.basis import Basis
from phd_experiments.torch_ode.torch_rk45 import TorchRK45


class TensorODEBLOCK(torch.nn.Module):
    NON_LINEARITIES = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid(), 'tanh': torch.nn.Tanh()}
    BASIS = ['None', 'poly', 'trig']

    def __init__(self, input_dimensions: List[int], output_dimensions: List[int],
                 tensor_dimensions: List[int], basis_str: str, t_span: Tuple, non_linearity: None | str = None,
                 t_eval: List = None):
        super().__init__()
        # FIXME add explicit params check
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions
        self.tensor_dimensions = tensor_dimensions
        self.non_linearity = non_linearity
        self.basis_str = basis_str
        self.t_span = t_span
        # assert parameters
        if non_linearity and non_linearity not in TensorODEBLOCK.NON_LINEARITIES.keys():
            raise ValueError(
                f'Non-linearity {self.non_linearity} not supported : must be one of '
                f'{TensorODEBLOCK.NON_LINEARITIES.keys()}')
        assert len(input_dimensions) == len(tensor_dimensions), f"For simplification we start with len(tensor_dims) = " \
                                                                f"len(input_dims) got len(input_dimensions) " \
                                                                f"= {len(input_dimensions)} while len(tensor_dims) " \
                                                                f"= {len(tensor_dimensions)}"
        assert isinstance(input_dimensions, list), "Input dimensions must be  a list"
        assert isinstance(output_dimensions, list), "Output dimensions must be  a list"
        assert isinstance(tensor_dimensions, list), "Tensor dimensions must be  a list"
        # parse basis function params
        basis_ = basis_str.split(',')
        assert basis_[0] in TensorODEBLOCK.BASIS, f"unknown basis {basis_[0]} : must be {TensorODEBLOCK.BASIS}"
        self.basis_fn = basis_[0]
        if basis_[0] == 'None':
            self.basis_params = None
        if basis_[0] == 'poly':
            self.basis_params = {'dim': int(basis_[1])}
        elif basis_[0] == 'trig':
            self.basis_params = {'a': basis_[1], 'b': basis_[2], 'c': basis_[3]}

        # C dimensions
        if self.basis_fn == 'None':
            C_dims = tensor_dimensions.copy()
            C_dims.extend(tensor_dimensions.copy())
            # C_dim.append(1) # time
        elif self.basis_fn == 'poly':
            D = int(np.prod(self.tensor_dimensions))
            assert isinstance(D, int)  # used as repeats
            C_dims = tensor_dimensions.copy()  # the output is the projected latent A
            C_dims.extend(list(np.repeat(a=int(int(self.basis_params['dim']) + 1), repeats=D + 1)))  # +1 for time
        elif self.basis_fn == 'trig':
            C_dims = tensor_dimensions.copy()
            C_dims.extend(tensor_dimensions.copy())
            C_dims.append(2)  # sin and cos

        # assert len(U_sizes) % 2 == 0, "U sizes must be odd"
        # F_dimension
        # Tensor param dims

        # P_dimensions
        P_dims = input_dimensions.copy()
        P_dims.extend(tensor_dimensions.copy())

        F_dims = tensor_dimensions.copy()
        F_dims.extend(output_dimensions.copy())
        self.P = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=P_dims),
                                    requires_grad=True)
        self.C = torch.nn.Parameter(
            torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=C_dims), requires_grad=False)
        self.F = torch.nn.Parameter(torch.distributions.Uniform(low=0.01, high=1.0).sample(sample_shape=F_dims),
                                    requires_grad=False)

        # Create solver
        assert t_span[0] < t_span[1], "t_span[0] must be < t_span[1]"
        if t_eval is not None:
            assert t_eval[0] >= t_span[0] and t_eval[1] <= t_span[1], "t_eval must be subset of t_span ranges"
        self.monitor = {'U': [self.C], 'P': [self.P], 'F': [self.F]}

    # TODO
    @staticmethod
    def params_check(input_dims: List[int], output_dims: List[int]):
        pass

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

        self.monitor['U'].append(torch.clone(self.C).detach())
        self.monitor['P'].append(torch.clone(self.P).detach())
        self.monitor['F'].append(torch.clone(self.F).detach())

        # param check
        assert len(x.size()) == 2, "No support for batch inputs with d>2 yet"

        # start forward pass
        z0 = x  # redundant but useful for convention convenience
        # assume z0 sizes/dimensions are batch x d0 x d1 x ... x d_z-1
        # assume P sizes/dimensions are d_z-1 x ... x d1 x d0 x d0 x d1 x ... x (d_A-1)
        z0_contract_dims = list(range(1, len(self.input_dimensions) + 1))
        P_contract_dims = list(range(len(self.input_dimensions)))
        A0 = torch.tensordot(a=z0, b=self.P, dims=(z0_contract_dims, P_contract_dims))
        torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32, )
        # a0 is a single sample in the Bx dim(A0) batch of samples

        # FIXME tried vmap but didn't work for batch solve
        # FIXME sol1 : see what torchdiffeq did in odeint function
        # FIXME sol2 : try vmap again , try to solve issues
        # FIXME vmap issue # 1: RuntimeError: Batching rule not implemented for aten::item.
        #  We could not generate a fallback.
        # FIXME vmap issue # 2 RuntimeError: Batching rule not implemented for aten::is_nonzero.
        #  We could not generate a fallback.
        # https://pytorch.org/tutorials/prototype/vmap_recipe.html#so-what-is-vmap
        # https://pytorch.org/functorch/stable/generated/functorch.vmap.html
        #  batched_solve = vmap(func=solve_, in_dims=0)
        solve_ = lambda a0: torch_solver.solve_ivp(func=self.ode_f, t_span=self.t_span, z0=a0,
                                                   args=(self.C, self.basis_fn, self.basis_params)).zf
        # FIXME loop apply : slow slow slow
        batch_size = A0.size()[0]
        A_f_list = [solve_(A0[b, :]) for b in range(batch_size)]
        A_f = torch.stack(A_f_list)
        A_contract_dims = list(range(1, len(self.tensor_dimensions) + 1))
        F_contract_dims = list(range(0, len(self.tensor_dimensions)))
        y_hat = torch.tensordot(a=A_f, b=self.F, dims=(A_contract_dims, F_contract_dims))
        return y_hat

    @staticmethod
    def tensor_contract(C: torch.Tensor, A_basis: torch.Tensor) -> torch.Tensor:
        A_basis_dims = A_basis.size()
        assert len(A_basis_dims) == 2, "Assume A basis dims = (poly_dim + 1) X (x_dim+1)"
        n_basis = list(A_basis.size())[1]
        c_n_dims = len(C.size())
        c_contract_dims = list(range(c_n_dims))[-n_basis:]
        assert len(c_contract_dims) == n_basis, "num of contract dims in must = num_basis"
        res = C
        res_contract_dim = list(range(len(C.size())))[-n_basis]
        for j in range(n_basis):
            res = torch.tensordot(a=res, b=A_basis[:, j], dims=([res_contract_dim], [0]))
        return res

    def ode_f(self, t: float, A: torch.Tensor, C: torch.Tensor, basis_fn: str, basis_params: dict):
        # TODO batched dot (what we need)
        # https://pytorch.org/tutorials/prototype/vmap_recipe.html
        """

        Parameters
        ----------
        t
        A : Batch x tensor dimensions
        C : Coeff tensor
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
        if basis_fn == 'None':
            A_contract_dims = list(range(len(A.size())))
            C_contract_dims = list(range(len(C.size())))[-len(A_contract_dims):]
            dAdt = torch.tensordot(a=C, b=A, dims=[C_contract_dims, A_contract_dims])
        elif basis_fn == 'poly':
            A_basis = Basis.poly(x=A, t=t, poly_dim=basis_params['dim'])
            dAdt = TensorODEBLOCK.tensor_contract(C,A_basis)
        elif basis_fn == 'trig':
            A_basis = Basis.trig(A, t, float(basis_params['a']), float(basis_params['b']), float(basis_params['c']))
            U_contract_dims = list(range(len(self.tensor_dimensions), len(C.size())))
            A_contract_dims = list(range(1, len(A_basis.size())))
            dAdt = torch.tensordot(A_basis, C, dims=(A_contract_dims, U_contract_dims))
        else:
            raise ValueError(f"basis_fn : {basis_fn} is not supported : must be {TensorODEBLOCK.BASIS}")

        return dAdt

        # if not non_linearity:
        #     dAdt = L
        # else:
        #     non_linearity_fn = TensorODEBLOCK.NON_LINEARITIES[non_linearity]
        #     dAdt = non_linearity_fn(L)
        # return dAdt
