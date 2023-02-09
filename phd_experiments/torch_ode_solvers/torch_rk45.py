"""
Based on
i) Solving Ordinary Differential Equations I - Nonstiff Problems (Second Revised Edition) :
    E. Hairer , S. P. Nørsett ,G. Wanner

Section: General Formulation of Runge-Kutta Methods  (p.134 eqn 1.8)

ii) The set of values in RK4(5) implementation of scipy ivp
https://github.com/scipy/scipy/blob/v1.9.2/scipy/integrate/_ivp/rk.py#L366

This is the first attempt to implement torch-based RK45 method to apply for trainable-tensor ODEs
"""
from typing import Callable, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from phd_experiments.torch_ode_solvers.common import torch_select_initial_step, torch_rms_norm
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchODESolver, TorchODESolverSolution


class TorchRK45(TorchODESolver):
    # from https://github.com/scipy/scipy/blob/v1.9.2/scipy/integrate/_ivp/rk.py#L366
    ORDER = 5
    ERROR_ESTIMATOR_ORDER = 4
    N_STAGES = 6

    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10

    def __init__(self, device: torch.device, tensor_dtype: torch.dtype, step_size: [float, str] = 0.01, rtol=1e-3,
                 atol=1e-6, is_batch: bool = True):
        super().__init__(step_size)
        self.is_batch = is_batch
        self.atol = atol
        self.rtol = rtol
        self.tensor_dtype = tensor_dtype
        self.K = None
        self.device = device
        self.C = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1], dtype=self.tensor_dtype, device=self.device)
        self.A = torch.tensor([
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
        ], dtype=self.tensor_dtype, device=self.device)
        self.B = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=self.tensor_dtype,
                              device=self.device)
        self.E = torch.tensor([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
                              dtype=self.tensor_dtype, device=self.device)

    def solve_ivp(self, func: Callable[[float, torch.Tensor, ...], torch.Tensor], t_span: Tuple,
                  z0: torch.Tensor, args: Tuple = None) -> TorchODESolverSolution:
        assert z0.dtype == self.tensor_dtype, f"Tensor must be of type {self.tensor_dtype}"
        # step adaptation to align tf correctly
        t0, tf = t_span
        # simplify func signature
        if args:
            func = lambda t, x, func=func: func(t, x, *args)
        # start integration
        z = z0.type(self.tensor_dtype)
        f = func(t0, z0)
        t = t0
        z_trajectory = [z0]
        t_values = [t0]
        K_sizes = [TorchRK45.N_STAGES + 1]
        K_sizes.extend(list(z0.size())[::-1])
        self.K = torch.empty(K_sizes, dtype=self.tensor_dtype, device=self.device)
        f0 = func(t0, z0)
        h = torch_select_initial_step(fun=func, t0=t0, y0=z0, f0=f0, direction=1, order=self.ERROR_ESTIMATOR_ORDER,
                                      rtol=self.rtol, atol=self.atol)
        finished = False
        while not finished:
            # try to make one step ahead

            z, f, h, t = TorchRK45._torch_rk_step_adaptive_step(func=func, t=t, tf=tf, z=z, f=f, h=h, A=self.A,
                                                                B=self.B, C=self.C, K=self.K, E=self.E, atol=self.atol,
                                                                rtol=self.rtol, is_batch=self.is_batch)

            if abs(t - tf) < 1e-4:
                finished = True
            z_trajectory.append(z)
            t_values.append(t)
        sol = TorchODESolverSolution(zf=z, z_trajectory=z_trajectory, t_values=t_values)
        return sol

    @staticmethod
    def _torch_rk_step(func: Callable, t: float, z: torch.Tensor, f: torch.Tensor, h: torch.Tensor,
                       A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, K: torch.Tensor, is_batch: bool) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # based on scipy integrate rk_step method
        # https://github.com/scipy/scipy/blob/v1.9.2/scipy/integrate/_ivp/rk.py#L14
        if is_batch:
            K[0] = f.T if is_batch else f
            for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
                dz = torch.matmul(K[:s].T, a[:s]) * h
                K[s] = func(t + c * h, z + dz).T if is_batch else func(t + c * h, z + dz)
            z_new = z + h * torch.matmul(K[:-1].T, B)
            f_new = func(t + h, z_new)
            K[-1] = f_new.T if is_batch else f_new
            return z_new, f_new

    @staticmethod
    def _torch_rk_step_adaptive_step(func: Callable, t: float, tf: float, z: torch.Tensor, f: torch.Tensor,
                                     h: float,
                                     A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, K: torch.Tensor,
                                     E: torch.Tensor, rtol: float,
                                     atol: float, is_batch: bool) -> tuple[Tensor, Tensor, float, float]:
        max_step = torch.inf
        min_step = 10 * np.abs(np.nextafter(t, np.inf) - t)
        if h > max_step:
            h = max_step
        elif h < min_step:
            h = min_step
        step_accepted = False
        step_rejected = False
        z_new = None
        f_new = None
        t_new = None
        while not step_accepted:
            if h < min_step:
                raise ValueError(f'h={h} < min_step = {min_step}. Cannot complete the integration, exiting!!!')
            t_new = t + h

            t_new = min(t_new, tf)

            h = t_new - t

            z_new, f_new = TorchRK45._torch_rk_step(func=func, t=t, z=z, f=f, h=h, A=A, B=B, C=C, K=K,
                                                    is_batch=is_batch)
            scale = atol + torch.maximum(torch.abs(z_new), torch.abs(z)) * rtol
            error_norm = TorchRK45._estimate_error_norm(K, E, h, scale)
            error_exponent = -1 / (TorchRK45.ERROR_ESTIMATOR_ORDER + 1)
            if error_norm < 1:
                if error_norm == 0:
                    factor = TorchRK45.MAX_FACTOR
                else:
                    factor = min(TorchRK45.MAX_FACTOR,
                                 TorchRK45.SAFETY * error_norm ** error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h *= factor

                step_accepted = True
            else:
                h *= max(TorchRK45.MIN_FACTOR,
                         TorchRK45.SAFETY * error_norm ** error_exponent)
                step_rejected = True
        return z_new, f_new, h, t_new

    @staticmethod
    def _estimate_error(K, E, h):
        return torch.matmul(K.T, E) * h

    @staticmethod
    def _estimate_error_norm(K, E, h, scale):
        return torch_rms_norm(TorchRK45._estimate_error(K, E, h) / scale)
