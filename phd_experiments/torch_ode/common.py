import logging

import numpy as np
import torch

EPS = np.finfo(float).eps


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_tol(rtol: float, atol: float, numel):
    logger = logging.getLogger()
    """Validate tolerance values."""

    if np.any(rtol < 100 * EPS):
        logger.warn("At least one element of `rtol` is too small. "
                    f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.")
        rtol = np.maximum(rtol, 100 * EPS)

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (numel,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


# copy from https://github.com/scipy/scipy/blob/v1.9.2/scipy/integrate/_ivp/common.py#L61
def torch_rms_norm(x:torch.Tensor):
    """Compute RMS norm."""
    return (torch.norm(x) / float(x.numel()) ** 0.5).item()


# copy from https://github.com/scipy/scipy/blob/v1.9.2/scipy/integrate/_ivp/common.py#L66
def torch_select_initial_step(fun: callable, t0: float, y0: torch.Tensor, f0: torch.Tensor, direction: int, order: int,
                              rtol: float, atol, *args):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.numel == 0:
        return np.inf

    scale = atol + torch.abs(y0) * rtol
    d0 = torch_rms_norm(y0 / scale)
    d1 = torch_rms_norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = torch_rms_norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)
