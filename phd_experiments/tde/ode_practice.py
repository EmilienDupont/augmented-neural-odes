import torch
from scipy.integrate import solve_ivp

from phd_experiments.tde.tensor_ode_solvers import TensorODESolver


def exponential_decay(t, y): return -0.5 * y


if __name__ == '__main__':
    tensor_ode_solver = TensorODESolver()
    y0 = torch.Tensor([[2, 4], [6, 8]])
    sol = tensor_ode_solver.solve(fun=exponential_decay, t_span=[0, 10], y0=y0, t_eval=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(sol.t)
    print(sol.y)
    print("")
