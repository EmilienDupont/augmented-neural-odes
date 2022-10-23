import numpy as np


def f_simple_linear_uncoupled_ode_1(t: float, y: np.ndarray, a: float):
    # http://web.math.ucsb.edu/~ebrahim/lin_ode_sys.pdf eqn (1)
    yprime = np.empty(2)
    yprime[0] = a * y[0]
    yprime[1] = -y[1]
    return yprime
