import logging

import numpy as np
import torch
from scipy.integrate import solve_ivp
from experiments_2.torch_ode.torch_euler import TorchEulerSolver
import pytest

from experiments_2.torch_ode.torch_rk45 import TorchRK45


@pytest.fixture(scope="session", autouse=True)
def test_setup():
    logging.basicConfig(level=logging.INFO)


def assert_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-3):
    return torch.norm(tensor1 - tensor2) <= eps


# Naive example
def test_torch_euler_1():
    def f(t: float, z: torch.Tensor):
        return z

    z0 = torch.tensor([1.0], dtype=TorchRK45.TORCH_DTYPE)
    h = 0.01
    zf = torch.tensor([16.0], dtype=TorchRK45.TORCH_DTYPE)
    euler_torch_ode_solver = TorchEulerSolver(step_size=h)
    sol = euler_torch_ode_solver.solve_ivp(func=f, t_span=(0.0, 4.0), z0=z0)
    assert_tensors(sol.zf, zf)


"""
Examples in 
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""


def test_torch_euler_2():
    def f(t: float, z: torch.Tensor):
        return -0.5 * z

    z0 = torch.Tensor([2, 4, 8])
    t_span = 0, 10
    zf = torch.tensor([0.01350781, 0.02701561, 0.05403123])
    h = 0.01
    euler_torch_ode_solver = TorchEulerSolver(step_size=h)
    sol = euler_torch_ode_solver.solve_ivp(func=f, t_span=t_span, z0=z0)
    assert_tensors(sol.zf, zf)


def test_torch_euler_3():
    # original example (ex. 3)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    a, b, c, d = (1.5, 1, 3, 1)
    t_span = 0, 15
    z0_vec = [10, 5]

    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        dzdt = [a * x - b * x * y, -c * y + d * x * y]
        return dzdt

    sol1 = solve_ivp(lotkavolterra, t_span, z0_vec, args=(a, b, c, d))
    zf = torch.tensor([sol1.y[0][-1], sol1.y[1][-1]])

    # start torch ode solver testing
    def f(t: float, z: torch.Tensor, *args):
        # https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation
        r = args[0]
        A = args[1]
        q = r + torch.matmul(A, z)
        dzdt = torch.mul(z, q)
        return dzdt

    z0 = torch.tensor(z0_vec)
    r = torch.tensor([a, -c], dtype=torch.float32)
    A = torch.tensor([[0, -b], [d, 0]], dtype=torch.float32)
    h = 0.001
    euler_torch_ode_solver = TorchEulerSolver(step_size=h)
    sol2 = euler_torch_ode_solver.solve_ivp(func=f, t_span=t_span, z0=z0, args=(r, A))
    assert_tensors(sol2.zf, zf)


def test_torch_rk45_1():
    def exponential_decay(t: float, y: torch.Tensor) -> torch.Tensor:
        return -0.5 * y

    z0_vec = [2, 4, 8]
    t_span = 0, 10
    sol_scipy = solve_ivp(fun=exponential_decay, t_span=t_span, y0=z0_vec, method='RK45')
    # get ground truth
    zf_actual_tensor = torch.tensor([sol_scipy.y[:, -1]])
    z0_tensor = torch.tensor([2, 4, 8],dtype=TorchRK45.TORCH_DTYPE)
    torch_rk45_solver = TorchRK45()

    sol_torch = torch_rk45_solver.solve_ivp(func=exponential_decay, z0=z0_tensor, t_span=t_span)
    assert_tensors(sol_torch.zf, zf_actual_tensor)


def test_torch_rk45_2():
    # run the basic example in
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a * x - b * x * y, -c * y + d * x * y]

    def lotkavolterra_generalized(t: float, z: torch.Tensor, *args) -> torch.Tensor:
        # https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation
        r = args[0]
        A = args[1]
        q = r + torch.matmul(A, z)
        dzdt = torch.mul(z, q)
        return dzdt

    z0_vec = [10, 5]
    t_span = 0, 10
    a, b, c, d = (1.5, 1, 3, 1)
    sol = solve_ivp(fun=lotkavolterra, t_span=t_span, y0=z0_vec, args=(a, b, c, d))
    # get ground truth
    zf_actual_tensor = torch.tensor(sol.y[:, -1], dtype=TorchRK45.TORCH_DTYPE)
    r = torch.tensor([a, -c], dtype=TorchRK45.TORCH_DTYPE)
    A = torch.tensor([[0, -b], [d, 0]], dtype=TorchRK45.TORCH_DTYPE)

    solver = TorchRK45()
    sol = solver.solve_ivp(func=lotkavolterra_generalized, t_span=t_span,
                           z0=torch.tensor(z0_vec, dtype=TorchRK45.TORCH_DTYPE),
                           args=(r, A))
    assert_tensors(zf_actual_tensor, sol.zf)


def test_torch_rk45_3():
    # https://nl.mathworks.com/help/matlab/ref/ode45.html
    # y`` = (A/B).t.y
    # y1` = y2
    # y2` = (A/B).t.y1
    def f_np(t: float, y: np.array, a: float, b: float) -> np.array:
        y_deriv = np.empty(2)
        y_deriv[0] = y[1]
        y_deriv[1] = float(a) / b * t * y[0]
        return y_deriv

    def f_tensor(t: float, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
        y_deriv = torch.empty(2)
        y_deriv[0] = y[1]
        y_deriv[1] = float(a) / b * t * y[0]
        return y_deriv

    a = 1
    b = 2
    t_span = 0, 5
    z0 = [0, 0.01]
    sol = solve_ivp(fun=f_np, t_span=t_span, y0=z0, args=(a, b))
    zf = torch.tensor(sol.y[:, -1], dtype=TorchRK45.TORCH_DTYPE)
    solver = TorchRK45()
    sol2 = solver.solve_ivp(func=f_tensor, t_span=t_span, z0=torch.tensor(z0, dtype=TorchRK45.TORCH_DTYPE), args=(a, b))
    assert_tensors(zf, sol2.zf)
