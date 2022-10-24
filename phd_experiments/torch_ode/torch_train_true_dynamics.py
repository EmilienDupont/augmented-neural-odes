import numpy as np


def f_ode_linear_uncoupled(t: float, y: np.ndarray, a: float) -> np.ndarray:
    # http://web.math.ucsb.edu/~ebrahim/lin_ode_sys.pdf eqn (1)
    yprime = np.empty(2)
    yprime[0] = a * y[0]
    yprime[1] = -y[1]
    return yprime


def f_ode_linear_coupled(t: float, y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    t : time
    y :
    A : matrix

    (Theory) http://web.math.ucsb.edu/~ebrahim/lin_ode_sys.pdf Sec. 3 p5
    (The Example) http://www.maths.surrey.ac.uk/explore/vithyaspages/coupled.html
    """
    dydt = np.matmul(A, y)
    return dydt


def f_van_der_pol(t: float, y: np.ndarray, mio: float) -> np.ndarray:
    # https://www.phys.uconn.edu/~rozman/Courses/P2200_13F/downloads/vanderpol/vanderpol-oscillator-draft.pdf
    # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    # Non-Linear time independent
    # https://www.johndcook.com/blog/2019/12/22/van-der-pol/
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = mio * (1 - y[0] ** 2) * y[1] - y[0]
    return dydt


def f_ode_non_linear_time_dependent(t: float, y: np.ndarray):
    pass
