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
    def poly(x: torch.Tensor, t: float, poly_dim: int) -> Tensor:
        """
        x : tensor of shape B x d1 x d2 x ... x dm : where m is the order of the tensor
        where B is batch size
         let D = d1xd2x...dm
        P : degrees of basis polynomial
        t : time value
        return  : Tensor of B x (x_dim+1) x (poly_dim) . x_dim + 1 because of time augmentation
        """
        # FIXME currently we assume that x is a batch of vectors i.e dim(x) = (batch X x_dim)
        assert len(x.size()) == 1, \
            "For now : we assume that the input to poly basis is a vector"
        B = list(x.size())[0]
        # t_tensor = torch.tensor(np.repeat(t, B), dtype=x.dtype).view(-1, 1)
        # x_flat = torch.flatten(x, start_dim=1)
        # # assert x_flat.requires_grad, "After flatten requires grad = False"
        x_aug = torch.cat(tensors=[x.T.view(1, -1), torch.tensor([t],dtype=x.dtype).view(1, 1)], dim=1)
        # x_pow = torch.clone(x_aug)
        x_poly_list = []
        for p in range(poly_dim + 1):
            x_pow = torch.pow(x_aug, p)
            x_poly_list.append(x_pow)
        x_basis = torch.concat(tensors=x_poly_list, dim=0)
        assert len(x_basis.size()) == 2, "For now : we assume that poly basis should return order-2 tensor" \
                                         "s.t. dim(x_basis) = (poly_dim+1) * (x_dim +1 ) "
        return x_basis

    @staticmethod
    def trig(x: torch.Tensor, t: float, a: float, b: float, c: float):
        n_x_dims = len(x.size())
        x_sin = a * torch.sin(b * x + c)
        x_cos = a * torch.cos(b * x + c)
        x_basis = torch.stack(tensors=[x_sin, x_cos], dim=n_x_dims)
        return x_basis

    # RBF
    """
    RBF http://www.scholarpedia.org/article/Radial_basis_function#Definition_of_the_method
    RBF function approximations : https://arxiv.org/pdf/1806.07705.pdf 
    RBF Practical Guide http://num.math.uni-goettingen.de/~schaback/teaching/sc.pdf
    NL Reg : RBF regression http://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/NonlinearRegression.pdf 
    """

    @staticmethod
    def rbf():
        pass

    # http://mlg.eng.cam.ac.uk/pub/pdf/Ras04.pdf
    @staticmethod
    # GP for ML http://gaussianprocess.org/gpml/chapters/RW.pdf
    # GP and TT https://arxiv.org/abs/1710.07324
    def GPKernel():
        pass
