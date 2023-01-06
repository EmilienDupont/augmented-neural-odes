import logging
from typing import List

import numpy as np
import torch
from scipy.integrate import solve_ivp
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
import pytest

from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

#######################
# Test Global Variables
# FIXME : make it better !!
#######################
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TENSOR_DTYPE = torch.float32


##########################

@pytest.fixture(scope="session", autouse=True)
def test_setup():
    logging.basicConfig(level=logging.INFO)


def assert_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-6):
    return torch.norm(tensor1 - tensor2) <= eps


# Naive example
def test_torch_euler_1():
    def f(t: float, z: torch.Tensor):
        return z

    z0 = torch.tensor([1.0], dtype=TENSOR_DTYPE, device=DEVICE)
    h = 0.01
    zf = torch.tensor([16.0], dtype=TENSOR_DTYPE, device=DEVICE)
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

    z0 = torch.tensor([2, 4, 8], dtype=TENSOR_DTYPE, device=DEVICE)
    t_span = 0, 10
    zf = torch.tensor([0.01350781, 0.02701561, 0.05403123], dtype=TENSOR_DTYPE, device=DEVICE)
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
    zf = torch.tensor([sol1.y[0][-1], sol1.y[1][-1]], device=DEVICE)

    # start torch ode solver testing
    def f(t: float, z: torch.Tensor, *args):
        # https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation
        r = args[0]
        A = args[1]
        q = r + torch.matmul(A, z)
        dzdt = torch.mul(z, q)
        return dzdt

    z0 = torch.tensor(z0_vec, device=DEVICE)
    r = torch.tensor([float(a), -c], device=DEVICE)
    A = torch.tensor([[0.0, -b], [d, 0]], device=DEVICE)
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
    zf_actual_tensor = torch.tensor([sol_scipy.y[:, -1]], device=DEVICE, dtype=TENSOR_DTYPE)
    z0_tensor = torch.tensor([2, 4, 8], device=DEVICE, dtype=TENSOR_DTYPE)
    torch_rk45_solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)

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
    zf_actual_tensor = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    r = torch.tensor([a, -c], dtype=TENSOR_DTYPE, device=DEVICE)
    A = torch.tensor([[0, -b], [d, 0]], dtype=TENSOR_DTYPE, device=DEVICE)

    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol = solver.solve_ivp(func=lotkavolterra_generalized, t_span=t_span,
                           z0=torch.tensor(z0_vec, dtype=TENSOR_DTYPE, device=DEVICE),
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
        y_deriv = torch.empty(2, dtype=TENSOR_DTYPE, device=DEVICE)
        y_deriv[0] = y[1]
        y_deriv[1] = float(a) / b * t * y[0]
        return y_deriv

    a = 1
    b = 2
    t_span = 0, 5
    z0 = [0, 0.01]
    sol = solve_ivp(fun=f_np, t_span=t_span, y0=z0, args=(a, b))
    zf = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol2 = solver.solve_ivp(func=f_tensor, t_span=t_span, z0=torch.tensor(z0, dtype=TENSOR_DTYPE, device=DEVICE),
                            args=(a, b))
    assert_tensors(zf, sol2.zf)


def test_rk45_4():
    # https://nl.mathworks.com/help/matlab/ref/ode45.html
    # y'=2t
    def func(t: float, y: np.ndarray):
        return 2 * t

    def func2(t: float, y: torch.Tensor):
        return torch.tensor(2 * t, dtype=TENSOR_DTYPE, device=DEVICE)

    # get ground truth
    z0 = [0]
    t_span = 0, 5
    sol = solve_ivp(fun=func, t_span=t_span, y0=z0)
    zf = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol2 = solver.solve_ivp(func=func2, t_span=t_span, z0=torch.tensor(z0, dtype=TENSOR_DTYPE, device=DEVICE))
    assert_tensors(zf, sol2.zf)


def test_rk45_5():
    # https://nl.mathworks.com/help/matlab/ref/ode45.html
    # van der Pol equation
    def func(t: float, y: np.ndarray, mio: float):
        dydt = np.empty(2)
        dydt[0] = y[1]
        dydt[1] = mio * (1 - y[0] ** 2) * y[1] - y[0]
        return dydt

    def func2(t: float, y: torch.Tensor, mio: float):
        dydt = torch.empty(2, device=DEVICE, dtype=TENSOR_DTYPE)
        dydt[0] = y[1]
        dydt[1] = mio * (1 - y[0] ** 2) * y[1] - y[0]
        return dydt

    z0 = [2, 0]
    t_span = 0, 20
    mio = 1
    sol = solve_ivp(fun=func, t_span=t_span, y0=z0, args=(mio,))
    zf = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol2 = solver.solve_ivp(func=func2, t_span=t_span, z0=torch.tensor(z0, dtype=TENSOR_DTYPE, device=DEVICE),
                            args=(mio,))
    assert_tensors(zf, sol2.zf)


def test_rk45_6():
    # https://nl.mathworks.com/help/matlab/ref/ode45.html
    # Solve ODE with Multiple Initial Conditions
    def f(t: float | torch.Tensor, y: np.ndarray | List | torch.Tensor):
        if isinstance(y, (np.ndarray, list)) and isinstance(t, float):
            dydt = -2 * y + 2 * np.cos(t) * np.sin(2 * t)
        elif isinstance(y, torch.Tensor):
            dydt = -2 * y + 2 * torch.cos(torch.tensor(t, dtype=TENSOR_DTYPE, device=DEVICE)) * \
                   torch.sin(torch.tensor(2 * t, dtype=TENSOR_DTYPE, device=DEVICE))
        else:
            raise ValueError(f'type of y is not known')
        return dydt

    z0 = list(np.arange(-5, 6))
    t_span = 0, 3
    sol = solve_ivp(fun=f, t_span=t_span, y0=z0)
    zf = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol2 = solver.solve_ivp(func=f, t_span=t_span, z0=torch.tensor(z0, dtype=TENSOR_DTYPE, device=DEVICE))
    assert_tensors(zf, sol2.zf)


def test_rk45_7():
    # https://nl.mathworks.com/help/matlab/ref/ode45.html#bu3l43b
    # ODE with time dependent terms
    """

    Returns
    -------

    """
    """"
    ft = linspace(0,5,25);
    f = ft.^2 - ft - 3;
    
    gt = linspace(1,6,25);
    g = 3*sin(gt-0.25);
    """
    t_f = np.arange(0, 25 + 1, 5)
    f = t_f ** 2 - t_f - 3
    t_g = np.arange(0, 25 + 1, 6)
    g = 3 * np.sin(t_g - 0.25)

    def func(t, y, t_f, f, t_g, g):
        """

        Parameters
        ----------
        t
        y
        t_f
        f
        t_g
        g

        Returns
        -------
        Matlab code
        function dydt = myode(t,y,ft,f,gt,g)
        f = interp1(ft,f,t); % Interpolate the data set (ft,f) at time t
        g = interp1(gt,g,t); % Interpolate the data set (gt,g) at time t
        dydt = -f.*y + g; % Evaluate ODE at time t

        """
        y_numpy = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        t_numpy = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        fval = np.interp(x=t_numpy, xp=t_f, fp=f)
        gval = np.interp(x=t_numpy, xp=t_g, fp=g)
        dydt = -fval * y_numpy + gval
        if isinstance(y, (np.ndarray, list)):
            return dydt
        elif isinstance(y, torch.Tensor):
            return torch.tensor(dydt, dtype=TENSOR_DTYPE, device=DEVICE)

    z0 = [1]
    t_span = 1, 5
    sol = solve_ivp(fun=func, t_span=t_span, y0=z0, args=(t_f, f, t_g, g))
    zf = torch.tensor(sol.y[:, -1], dtype=TENSOR_DTYPE, device=DEVICE)
    solver = TorchRK45(device=DEVICE, tensor_dtype=TENSOR_DTYPE)
    sol2 = solver.solve_ivp(func=func, t_span=t_span, z0=torch.tensor(z0, dtype=TENSOR_DTYPE, device=DEVICE),
                            args=(t_f, f, t_g, g))
    assert_tensors(zf, sol2.zf)
