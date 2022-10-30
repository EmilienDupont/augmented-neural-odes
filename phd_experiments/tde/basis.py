"""
Basis functions used to encode input before apply tensors to it
"""
from typing import List

import numpy as np
import torch
from torch import Tensor


class Basis:
    """
    References
    https://arxiv.org/pdf/1405.5713.pdf
    https://proceedings.neurips.cc/paper/2016/file/5314b9674c86e3f9d1ba25ef9bb32895-Paper.pdf  sec 2
    https://arxiv.org/pdf/2111.14540.pdf

    """

    def __init__(self):
        pass

    @staticmethod
    def poly(x: torch.Tensor,poly_dim: int) -> Tensor:
        """
        x : tensor of shape B x d1 x d2 x ... x dm : where m is the order of the tensor
        where B is batch size
         let D = d1xd2x...dm
        P : degrees of basis polynomial
        t : time value
        return  : Tensor of B x (D+1) x (P+1)
        """

        B = list(x.size())[0]
        # t_tensor = torch.tensor(np.repeat(t, B), dtype=x.dtype).view(-1, 1)
        x_flat = torch.flatten(x, start_dim=1)
        # assert x_flat.requires_grad, "After flatten requires grad = False"
        # x_aug = torch.cat(tensors=[x_flat], dim=1)
        x_pow = torch.clone(x_flat)
        x_poly_list = [x_pow]  # to the power 0
        for p in range(1, poly_dim + 1):
            x_pow = torch.mul(x_pow, x_flat)
            x_poly_list.append(x_pow)
        x_basis = torch.stack(tensors=x_poly_list, dim=2)
        return x_basis
