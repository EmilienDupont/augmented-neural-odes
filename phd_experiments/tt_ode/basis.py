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
    https://github.com/goroda/PyTorchPoly
    https://mlatcl.github.io/deepnn/background/background-basis-functions.html
    https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
    """

    def __init__(self):
        pass

    @staticmethod
    def poly(x: torch.Tensor, t: [float|None], poly_deg: int) -> List[Tensor]:
        """
        x : tensor of shape B x Dx
        P : degrees of basis polynomial
        t : time value # FIXME ignored for now
        return  : Tensor of List of tensor of shape Bx(poly_dim + 1)
        """
        """
        https://github.com/goroda/PyTorchPoly 
        """
        """
            x : torch.Tensor of size Batch x dim
            """
        assert isinstance(x, torch.Tensor)
        assert len(x.size()) == 2, "Supporting poly-basis generation for only torch-vectors"
        B = x.size()[0]
        x = torch.cat([x, torch.tensor(t, dtype=x.dtype).repeat(B).view(-1, 1)], dim=1) if t is not None else x
        pow_tensors = [torch.ones(x.size()), x] # to the power 0 and 1
        for deg in range(2, poly_deg + 1):
            pow_tensors.append(torch.pow(x, deg))
        Phi = list(torch.permute(input=torch.stack(tensors=pow_tensors, dim=0), dims=[2, 1, 0]))
        return Phi

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
        # IMP : http://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/NonlinearRegression.pdf
        # https://medium.com/analytics-vidhya/nonlinear-regression-tutorial-with-radial-basis-functions-cdb7650104e7
        # https://www.chadvernon.com/blog/rbf/
        # https://elcorto.github.io/pwtools/written/background/rbf.html
        pass

    # http://mlg.eng.cam.ac.uk/pub/pdf/Ras04.pdf
    @staticmethod
    # GP for ML http://gaussianprocess.org/gpml/chapters/RW.pdf
    # GP and TT https://arxiv.org/abs/1710.07324
    def GPKernel():
        pass
