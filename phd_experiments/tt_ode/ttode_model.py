from typing import List, Tuple, Callable, Union
import numpy as np
import torch
from torchdiffeq import odeint
from anode.models import ODEFunc
from phd_experiments.tn.tt import TensorTrainFixedRank
from torch import Tensor
from phd_experiments.tt_ode.basis import Basis
from phd_experiments.tt_ode.ttode_als import TTOdeAls, TensorTrainContainer
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


class TerminalNeuralNetwork(torch.nn.Module):
    NON_LINEARITIES = ["relu", "sigmoid", "softplus"]

    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_dim, hidden_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def norm(self) -> float:
        tot_norm = 0
        num_layers = len(self.model)
        for layer_idx in range(num_layers):
            if hasattr(self.model[layer_idx],'weight') and isinstance(self.model[layer_idx].weight,Tensor):
                tot_norm+=self.model[layer_idx].weight.norm()

        return tot_norm


class TensorTrainODEBLOCK(torch.nn.Module):
    NON_LINEARITIES = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid(), 'tanh': torch.nn.Tanh()}
    BASIS = ['None', 'poly', 'trig']
    FORWARD_IMPL = ['ttode_als', 'mytorch', 'torchdiffeq', 'nn']

    def __init__(self, input_dimensions: List[int], output_dimensions: List[int],
                 tensor_dimensions: List[int], basis_str: str, t_span: Tuple, non_linearity: None | str = None,
                 t_eval: List = None, forward_impl_method: str = "batch_torch",
                 tensor_dtype: torch.dtype = torch.float32, tt_rank=Union[int | list[int]]):
        super().__init__()
        # FIXME add explicit params check
        self.tensor_dtype = tensor_dtype
        self.forward_impl_method = forward_impl_method
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions
        self.tensor_dimensions = tensor_dimensions
        self.non_linearity = non_linearity
        self.basis_str = basis_str
        self.t_span = t_span
        self.tt_rank = tt_rank
        self.nfe = 0
        self.tt_container = TensorTrainContainer()
        # assert parameters
        assert len(input_dimensions) == 1, " Supporting input vectors only"
        assert len(output_dimensions) == 1, " Supporting output vectors only"
        assert len(tensor_dimensions) == 1, "Supporting vector higher dimension latents only"
        if non_linearity and non_linearity not in TensorTrainODEBLOCK.NON_LINEARITIES.keys():
            raise ValueError(
                f'Non-linearity {self.non_linearity} not supported : must be one of '
                f'{TensorTrainODEBLOCK.NON_LINEARITIES.keys()}')
        assert len(input_dimensions) == len(tensor_dimensions), f"For simplification we start with len(tensor_dims) = " \
                                                                f"len(input_dims) got len(input_dimensions) " \
                                                                f"= {len(input_dimensions)} while len(tensor_dims) " \
                                                                f"= {len(tensor_dimensions)}"
        assert isinstance(input_dimensions, list), "Input dimensions must be  a list"
        assert isinstance(output_dimensions, list), "Output dimensions must be  a list"
        assert isinstance(tensor_dimensions, list), "Tensor dimensions must be  a list"
        assert forward_impl_method in TensorTrainODEBLOCK.FORWARD_IMPL, f"forward_impl = {self.forward_impl_method} " \
                                                                        f"not supported , " \
                                                                        f"must be one of {TensorTrainODEBLOCK.FORWARD_IMPL}"

        # add is_batch flag

        # parse basis function params
        basis_ = basis_str.split(',')
        assert basis_[
                   0] in TensorTrainODEBLOCK.BASIS, f"unknown basis {basis_[0]} : must be {TensorTrainODEBLOCK.BASIS}"
        self.basis_fn = basis_[0]
        if basis_[0] == 'None':
            self.basis_params = None
        if basis_[0] == 'poly':
            self.basis_params = {'deg': int(basis_[1])}
        elif basis_[0] == 'trig':
            self.basis_params = {'a': basis_[1], 'b': basis_[2], 'c': basis_[3]}

        # W dimensions
        if self.basis_fn == 'None':
            W_dims = tensor_dimensions.copy()
            W_dims.extend(tensor_dimensions.copy())
            # C_dim.append(1) # time
        elif self.basis_fn == 'poly':
            # assume inputs / outputs are vectors (batches of vectors)

            W_dims = tensor_dimensions.copy()  # the output is the projected latent A
            W_dims.extend(list(np.repeat(a=int(int(self.basis_params['deg'] + 1)),  # +1 for const-term
                                         repeats=self.tensor_dimensions[0] + 1)))  # +1 for time
        elif self.basis_fn == 'trig':
            W_dims = tensor_dimensions.copy()
            W_dims.extend(tensor_dimensions.copy())
            W_dims.append(2)  # sin and cos

        P_dims = input_dimensions.copy()
        P_dims.extend(tensor_dimensions.copy())
        assert len(P_dims) == 2, "No support for the projection tensor P with n_dims > 2 , yet !"

        # F_dims = tensor_dimensions.copy()
        # F_dims.extend(output_dimensions.copy())

        # initialize model parameters
        ulow, uhigh = 1e-7, 1e-5
        self.P = torch.nn.Parameter(torch.distributions.Uniform(low=ulow, high=uhigh).sample(sample_shape=P_dims))

        # self.W = torch.nn.Parameter(
        #     torch.distributions.Uniform(low=ulow, high=uhigh).sample(sample_shape=W_dims))
        D_a = tensor_dimensions[0]  # assume tensor dimensions is for length 1 : i.e. project vector to vector
        # TODO support list of ranks or adaptive using ALS / DMRG ??
        assert isinstance(tt_rank, int), "Now only supported fixed-rank TT"
        # FIXME TT structure assume poly basis fun
        if self.forward_impl_method == 'ttode_als':
            self.W = TensorTrainFixedRank(order=D_a + 1, core_input_dim=self.basis_params['deg'] + 1, out_dim=D_a,
                                          fixed_rank=self.tt_rank, requires_grad=False)  # not optimizable by grad
            self.P = torch.nn.Parameter(torch.distributions.Uniform(low=ulow, high=uhigh).sample(sample_shape=P_dims),
                                        requires_grad=False)
        else:
            self.W = TensorTrainFixedRank(order=D_a + 1, core_input_dim=self.basis_params['deg'] + 1, out_dim=D_a,
                                          fixed_rank=self.tt_rank, requires_grad=True)  # optimizable by grad
            self.P = torch.nn.Parameter(torch.distributions.Uniform(low=ulow, high=uhigh).sample(sample_shape=P_dims),
                                        requires_grad=True)
            ####
        self.terminal_nn = TerminalNeuralNetwork(input_dim=self.tensor_dimensions[0], out_dim=self.output_dimensions[0],
                                                 hidden_dim=256)

        # Create solver
        assert t_span[0] < t_span[1], "t_span[0] must be < t_span[1]"
        if t_eval is not None:
            assert t_eval[0] >= t_span[0] and t_eval[1] <= t_span[1], "t_eval must be subset of t_span ranges"
        self.monitor = {'W': [self.W], 'P': [self.P], 'F': [self.terminal_nn]}

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : batch of vectors (order-2 tensor / matrix) with dims = batch X x_dims
        t_span
        t_eval

        Returns
        -------

        """
        # Record parameters for monitoring

        # self.monitor['M'].append(torch.clone(self.M).detach())
        # self.monitor['P'].append(torch.clone(self.P).detach())
        # self.monitor['F'].append(torch.clone(self.F).detach())

        # param check
        assert len(x.size()) == 2, "No support for batch inputs with d>1 yet"
        y_hat = self.forward_impl(x)
        return y_hat

    def forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        zf = None
        # start forward pass
        z0_contract_dims = list(range(1, len(self.input_dimensions) + 1))
        P_contract_dims = list(range(len(self.input_dimensions)))
        z0 = torch.tensordot(a=x, b=self.P, dims=(z0_contract_dims, P_contract_dims))
        if self.forward_impl_method == 'ttode_als':
            tt_ode_alias = TTOdeAls.apply
            zf = tt_ode_alias(x, self.P, self.input_dimensions, self.W, self.tt_container, self.tensor_dtype,
                              self.tt_ode_func, self.t_span
                              , self.basis_fn, self.basis_params)
        elif self.forward_impl_method == 'gen_linear_const':
            """
            https://www.stat.uchicago.edu/~lekheng/work/mcsc2.pdf 
            https://en.wikipedia.org/wiki/Matrix_differential_equation#:~:text=A%20matrix%20differential%20equation%20contains,the%20functions%20to%20their%20derivatives. 
            https://people.math.wisc.edu/~angenent/519.2016s/notes/linear-systems-homogeneous.html
            """
            pass
        elif self.forward_impl_method == 'mytorch':
            torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=self.tensor_dtype, is_batch=True)
            zf = torch_solver.solve_ivp(func=self.tt_ode_func, t_span=self.t_span, z0=z0,
                                        args=(self.W, self.basis_fn, self.basis_params)).zf

        # this if branch is just left for verifying results when needed (verify my modification for batch rk45)
        elif self.forward_impl_method == 'single_torch':
            # A0 is a single sample in the Bx dim(A0) batch of samples
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

            batch_size = z0.size()[0]
            torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=self.tensor_dtype, is_batch=True)
            solve_ = lambda a0: torch_solver.solve_ivp(func=self.tt_ode_func, t_span=self.t_span, z0=a0,
                                                       args=(self.W, self.basis_fn, self.basis_params)).zf
            # FIXME Warning ! slow slow slow
            zf = torch.stack([solve_(z0[b, :]) for b in range(batch_size)], dim=0)

        elif self.forward_impl_method == 'torchdiffeq':
            func_ = lambda t, A: self.tt_ode_func(t, A, self.W, basis_fn='poly', basis_params=self.basis_params)
            zf = odeint(func=func_, t=torch.tensor([0.0, 1.0]), y0=z0, rtol=1e-5, atol=1e-8)[-1, :]
        elif self.forward_impl_method == 'nn':
            nn_ode_func = ODEFunc(device=torch.device('cpu'), data_dim=self.tensor_dimensions[0], hidden_dim=64)
            func_ = lambda t, A: nn_ode_func(t, A)
            zf = odeint(func=func_, t=torch.tensor([0.0, 1.0]), y0=z0, rtol=1e-5, atol=1e-8)[-1, :]
        else:
            raise ValueError(f"forward_impl_method {self.forward_impl_method} not supported, "
                             f"must be one of {TensorTrainODEBLOCK.FORWARD_IMPL}")
        y_hat = self.terminal_nn(zf)
        return y_hat

    def tt_ode_func(self, t: float, z: torch.Tensor, W: TensorTrainFixedRank, basis_fn: str, basis_params: dict):
        # TODO batched dot (what we need)
        # https://pytorch.org/tutorials/prototype/vmap_recipe.html
        """

        Parameters
        ----------
        t
        z : Batch x tensor dimensions
        W : Coeff tensor
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
        self.nfe += 1  # count number of derivative function evaluations
        if basis_fn == 'None':
            if self.forward_impl_method == 'batch_torch':
                A_contract_dims = list(range(len(z.size())))[1:]
                W_contract_dims = list(range(len(W.size())))[-len(A_contract_dims):]
            elif self.forward_impl_method == 'single_torch':
                A_contract_dims = list(range(len(z.size())))
                W_contract_dims = list(range(len(W.size())))[-len(A_contract_dims):]

            else:
                raise ValueError(f'forward_impl_method {self.forward_impl_methfod} is not supported, must be one of '
                                 f'= {TensorTrainODEBLOCK.FORWARD_IMPL}')
            dzdt = torch.tensordot(a=z, b=W, dims=[A_contract_dims, W_contract_dims])

        elif basis_fn == 'poly':
            Phi = Basis.poly(x=z, t=t, poly_deg=basis_params['deg'])
            dzdt = W.contract_basis(basis_tensors=Phi)
        elif basis_fn == 'trig':
            Phi = Basis.trig(z, t, float(basis_params['a']), float(basis_params['b']), float(basis_params['c']))
            W_contract_dims = list(range(len(self.tensor_dimensions), len(W.size())))
            A_contract_dims = list(range(1, len(Phi.size())))
            dzdt = torch.tensordot(Phi, W, dims=(A_contract_dims, W_contract_dims))
        else:
            raise ValueError(f"basis_fn : {basis_fn} is not supported : must be {TensorTrainODEBLOCK.BASIS}")
        return dzdt

    def get_nfe(self):
        return self.nfe

    def get_terminal_nn(self):
        return self.terminal_nn

    def get_P(self):
        return self.P

    def get_W(self):
        return self.W
