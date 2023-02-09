# MIT License
#
# Copyright (c) 2021 Jacob Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Uniform and non-uniform probabilistic MPS classes"""
from math import sqrt, pi
from functools import partial
from typing import Union, Optional, Callable

import torch
from torch import nn

from .utils2 import einsum


class DataDomain:
    r"""
    Defines a domain for input data to a probabilistic model

    DataDomain supports both continuous and discrete domains, with the
    latter always associated with indices of the form `0, 1, ..., max_val-1`.
    For continuous domains, real intervals of the form `[min_val, max_val]`
    can be defined.

    Args:
        continuous (bool): Whether data domain is continuous or discrete
        max_val (int or float): For discrete domains, this is the number of
            indices to use, with the maximum index being max_val - 1. For
            continuous domains, this is the endpoint of the real interval.
        min_val (float): Only used for continuous domains, this is the
            startpoint of the real interval.
    """

    def __init__(
        self,
        continuous: bool,
        max_val: Union[int, float],
        min_val: Optional[float] = None,
    ):
        # Check defining input for correctness
        if continuous:
            assert max_val > min_val
            self.min_val = min_val
        else:
            assert max_val >= 0

        self.max_val = max_val
        self.continuous = continuous


class TrainableEmbedding(nn.Module):
    r"""
    Framework for trainable embedding function converting data to vectors

    This acts as a wrapper for a user-specified `torch.nn.Module` instance,
    whose parameters are trained jointly with those of the MPS using it.

    Args:
        emb_fun (torch.nn.Module): Initialized function arbitrary tensors of
            values and returning tensor of embedded vectors, which has one
            additional axis in the last position. These values must be either
            integers, for discrete data domains, or reals, for continuous
            data domains.
        data_domain (DataDomain): Object which specifies the domain on which
            the data fed to the embedding function is defined.
    """

    def __init__(self, emb_fun: Callable, data_domain: DataDomain):
        super().__init__()
        assert isinstance(emb_fun, nn.Module)

        # Save defining data
        self.domain = data_domain
        # Need to add on singleton dimension to agree with nn.Module format
        self.emb_fun = lambda x: emb_fun(x[..., None])
        # self.emb_fun = emb_fun

    def make_lambda(self, num_points: int = 1000):
        """
        Compute the lambda matrix used for normalization
        """
        # Compute the raw lambda matrix, computing number of points if needed
        if self.domain.continuous:
            points = torch.linspace(
                self.domain.min_val, self.domain.max_val, steps=num_points
            )
            self.num_points = num_points
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == num_points

            # Get rank-1 matrices for each point, then numerically integrate
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.trapz(emb_mats, points, dim=0)

        else:
            points = torch.arange(self.domain.max_val).long()
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == self.domain.max_val

            # Get rank-1 matrices for each point, then sum together
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.sum(emb_mats, dim=0)

        assert lamb_mat.ndim == 2
        assert lamb_mat.shape[0] == lamb_mat.shape[1]
        self.lamb_mat = lamb_mat

        # Check if the computed matrix is diagonal or multiple of the identity
        if torch.allclose(lamb_mat.diag().diag(), lamb_mat):
            lamb_mat = lamb_mat.diag()
            if torch.allclose(lamb_mat.mean(), lamb_mat):
                self.lamb_mat = lamb_mat.mean()
            else:
                self.lamb_mat = lamb_mat

    def forward(self, input_data):
        """
        Embed input data via the user-specified embedding function
        """
        self.make_lambda()
        return self.emb_fun(input_data)


class FixedEmbedding(nn.Module):
    r"""
    Framework for fixed embedding function converting data to vectors

    Args:
        emb_fun (function): Function taking arbitrary tensors of values and
            returning tensor of embedded vectors, which has one additional
            axis in the last position. These values must be either integers,
            for discrete data domains, or reals, for continuous data domains.
        data_domain (DataDomain): Object which specifies the domain on which
            the data fed to the embedding function is defined.
    """

    def __init__(self, emb_fun: Callable, data_domain: DataDomain):
        super().__init__()
        assert hasattr(emb_fun, "__call__")

        # Save defining data, compute lambda matrix
        self.domain = data_domain
        self.emb_fun = emb_fun
        self.make_lambda()

    @torch.no_grad()
    def make_lambda(self, num_points: int = 1000):
        """
        Compute the lambda matrix used for normalization
        """
        # Compute the raw lambda matrix, computing number of points if needed
        if self.domain.continuous:
            points = torch.linspace(
                self.domain.min_val, self.domain.max_val, steps=num_points
            )
            self.num_points = num_points
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == num_points

            # Get rank-1 matrices for each point, then numerically integrate
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.trapz(emb_mats, points, dim=0)

        else:
            points = torch.arange(self.domain.max_val).long()
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == self.domain.max_val

            # Get rank-1 matrices for each point, then sum together
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            lamb_mat = torch.sum(emb_mats, dim=0)

        assert lamb_mat.ndim == 2
        assert lamb_mat.shape[0] == lamb_mat.shape[1]
        self.lamb_mat = lamb_mat

        # Check if the computed matrix is diagonal or multiple of the identity
        if torch.allclose(lamb_mat.diag().diag(), lamb_mat):
            lamb_mat = lamb_mat.diag()
            if torch.allclose(lamb_mat.mean(), lamb_mat):
                self.lamb_mat = lamb_mat.mean()
            else:
                self.lamb_mat = lamb_mat

    def forward(self, input_data):
        """
        Embed input data via the user-specified embedding function
        """
        return self.emb_fun(input_data)


unit_interval = DataDomain(continuous=True, max_val=1, min_val=0)


def onehot_embed(tensor, emb_dim):
    """
    Function giving trivial one-hot embedding of categorical data
    """
    shape = tensor.shape + (emb_dim,)
    output = torch.zeros(*shape)
    output.scatter_(-1, tensor[..., None], 1)
    return output


def trig_embed(data, emb_dim=2):
    r"""
    Function giving embedding from powers of sine and cosine

    Based on Equation B4 of E.M. Stoudenmire and D.J. Schwab, "Supervised
    Learning With Quantum-Inspired Tensor Networks", NIPS 2016, which maps an
    input x in the unit interval to a d-dim vector whose j'th component
    (where j = 0, 1, ..., d-1) is:

    .. math::
        \phi(x)_j = \sqrt{d-1 \choose j} \cos(\frac{pi}{2}x)^{d-j-1}
        \sin(\frac{pi}{2}x)^{j}

    Written by Raphaëlle Tihon
    """
    from scipy.special import binom

    emb_data = []
    for s in range(emb_dim):
        comp = (
            torch.cos(data * pi / 2) ** (emb_dim - s - 1)
            * torch.sin(data * pi / 2) ** s
        )
        comp *= sqrt(binom(emb_dim - 1, s))
        emb_data.append(comp)
    emb_data = torch.stack(emb_data, dim=-1)
    assert emb_data.shape == data.shape + (emb_dim,)
    return emb_data


def init_mlp_embed(output_dim, num_layers=2, hidden_dims=[100], data_domain=None):
    """
    Initialize multilayer perceptron embedding acting on scalar inputs

    Args:
        output_dim: Dimensionality of the embedded output vectors
        num_layers: Total number of layers in the MLP embedding function
        hidden_dims: List of dimensions of the hidden layers for the MLP
        data_domain: If scalar inputs are not on the unit interval, a custom
            data domain must be specified to allow correct normalization
    """
    # Put all layer dimensions in single list
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims] * (num_layers - 1)
    assert len(hidden_dims) == num_layers - 1
    all_dims = [1] + list(hidden_dims) + [output_dim]

    # Create underlying MLP from list of nn Modules
    mod_list = []
    for i in range(num_layers):
        mod_list.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        mod_list.append(nn.Sigmoid())  # Sigmoid avoids putting outputs to 0
    emb_fun = nn.Sequential(*mod_list)

    # Initialize and return the trainable embedding function
    if data_domain is None:
        data_domain = unit_interval
    return TrainableEmbedding(emb_fun, data_domain)


sincos_embed = partial(trig_embed, emb_dim=2)
