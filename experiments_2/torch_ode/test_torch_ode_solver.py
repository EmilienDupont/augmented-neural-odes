import logging

import torch
from scipy.integrate import solve_ivp
from experiments_2.torch_ode.torch_euler import TorchEulerSolver
import pytest


@pytest.fixture(scope="session", autouse=True)
def test_setup():
    logging.basicConfig(level=logging.INFO)


def assert_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-3):
    return torch.norm(tensor1 - tensor2) <= eps


# Naive example
def test_torch_euler_1():
    def f(t: float, z: torch.Tensor):
        return z

    z0 = torch.Tensor([1.0])
    h = 0.01
    zf = torch.tensor([16.0])
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
    sol2 = euler_torch_ode_solver.solve_ivp(f, t_span, z0, r, A)
    assert_tensors(sol2.zf, zf)
