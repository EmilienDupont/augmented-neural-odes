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
from math import sqrt
from typing import Optional, Callable

import torch
from torch import Tensor, nn

from torchmps.mps_base import (
    contract_matseq,
    near_eye_init,
    normal_init,
    get_mat_slices,
    get_log_norm,
    slim_eval_fun,
)
from torchmps.utils2 import phaseify
from torchmps.embeddings import DataDomain, FixedEmbedding, TrainableEmbedding


class ProbMPS(nn.Module):
    r"""
    Fixed-length MPS model using L2 probabilities for generative modeling

    Probabilities of fixed-length inputs are obtained via the Born rule of
    quantum mechanics, making ProbMPS a "Born machine" model. For a model
    acting on length-n inputs, the probability assigned to the sequence
    :math:`x = x_1 x_2 \dots x_n` is :math:`P(x) = |h_n^T \omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A_t[x_t] + B) h_{t-1},

    with :math:`h_0 := \alpha` (for :math:`\alpha, \omega` trainable
    parameter vectors), :math:`A_t[i]` the i'th matrix slice of a
    third-order core tensor for the t'th input, and :math:`B` an optional
    bias matrix.

    Note that calling a :class:`ProbMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbMPS`.

    Args:
        seq_len: Length of fixed-length discrete sequence inputs. Inputs
            can be either batches of discrete sequences, with a shape of
            `(input_len, batch)`, or batches of vector sequences, with a
            shape of `(input_len, batch, input_dim)`.
        input_dim: Dimension of the inputs to each core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking adjacent MPS cores,
            which are assumed to be equal everywhere.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
        init_method: String specifying how to initialize the MPS core tensors.
            Giving "near_eye" initializes all core slices to near the identity,
            while "normal" has all core elements be normally distributed.
            Default: ``"near_eye"``
        embed_fun: Function which embeds discrete or continous scalar values
            into vectors of dimension `input_dim`. Must be able to take in a
            tensor of any order `n` and output a tensor of order `n+1`, where
            the scalar values of the input are represented as vectors in the
            *last* axis of the output.
            Default: ``None`` (no embedding function)
        domain: Instance of the `DataDomain` class, which specifies whether
            the input data domain is continuous vs. discrete, and what range
            of values the domain takes.
            Default: ``None``
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        bond_dim: int,
        complex_params: bool = False,
        use_bias: bool = False,
        init_method: str = "near_eye",
        embed_fun: Optional[Callable] = None,
        domain: Optional[DataDomain] = None,
    ) -> None:
        super().__init__()
        assert min(seq_len, input_dim, bond_dim) > 0

        # Initialize core tensor and edge vectors
        assert init_method in ("near_eye", "normal")
        init_fun = near_eye_init if init_method == "near_eye" else normal_init
        core_tensors = init_fun(
            (seq_len, input_dim, bond_dim, bond_dim), is_complex=complex_params
        )
        # Left and right vectors initialized to be identical, since it avoids
        # issues with exponentially small overlap
        rand_vec = torch.randn(bond_dim) / sqrt(bond_dim)
        edge_vecs = torch.stack((rand_vec,) * 2)
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)

        # Initialize (optional) bias matrices at zero
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)

        # Set other MPS attributes
        self.complex_params = complex_params
        self.embedding = None

        # Set up embedding object if desired
        if isinstance(embed_fun, (FixedEmbedding, TrainableEmbedding)):
            self.embedding = embed_fun
            if hasattr(embed_fun, "emb_dim"):
                assert self.embedding.emb_dim == input_dim
        elif embed_fun is not None:
            assert domain is not None
            self.embedding = FixedEmbedding(embed_fun, domain)
            assert self.embedding.emb_dim == input_dim

    def forward(
        self, input_data: Tensor, slim_eval: bool = False, parallel_eval: bool = False
    ) -> Tensor:
        """
        Get the log probabilities of batch of input data

        Args:
            input_data: Sequential with shape `(batch, seq_len)`, for
                discrete inputs, or shape `(batch, seq_len, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            log_probs: Vector with shape `(batch,)` giving the natural
                logarithm of the probability of each input sequence.
        """
        # Apply embedding function if it is defined
        if self.embedding is not None:
            input_data = self.embedding(input_data)

        if slim_eval:
            if self.use_bias:
                raise ValueError("Bias matrices not supported for slim_eval")
            psi_vals, log_scales = slim_eval_fun(
                input_data, self.core_tensors, self.edge_vecs
            )
        else:
            # Contract inputs with core tensors and add bias matrices
            mat_slices = get_mat_slices(input_data, self.core_tensors)
            if self.use_bias:
                mat_slices = mat_slices + self.bias_mat[None, None]

            #  Contract all bond dims to get (unnormalized) prob amplitudes
            psi_vals, log_scales = contract_matseq(
                mat_slices,
                self.edge_vecs[0],
                self.edge_vecs[1],
                parallel_eval,
                log_format=True,
            )

        # Get log normalization and check for infinities
        log_norm = self.log_norm()
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())

        # Compute unnormalized log probabilities
        log_uprobs = torch.log(torch.abs(psi_vals)) + log_scales

        # Return normalized probabilities
        return 2 * log_uprobs - log_norm

    def loss(
        self, input_data: Tensor, slim_eval: bool = False, parallel_eval: bool = False
    ) -> Tensor:
        """
        Get the negative log likelihood loss for batch of input data

        Args:
            input_data: Sequential with shape `(seq_len, batch)`, for
                discrete inputs, or shape `(seq_len, batch, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            loss_val: Scalar value giving average of the negative log
                likelihood loss of all sequences in input batch.
        """
        return -torch.mean(
            self.forward(input_data, slim_eval=slim_eval, parallel_eval=parallel_eval)
        )

    def log_norm(self) -> Tensor:
        r"""
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\log(|\psi|^2)`,
        where :math:`\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        # Account for bias matrices before calling log norm implementation
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None, None]
        else:
            core_tensors = self.core_tensors

        # Account for non-trivial lambda function in the embedding
        lamb_mat = None if self.embedding is None else self.embedding.lamb_mat

        return get_log_norm(core_tensors, self.edge_vecs, lamb_mat=lamb_mat)

    @property
    def seq_len(self):
        return self.core_tensors.shape[0]

    @property
    def input_dim(self):
        return self.core_tensors.shape[1]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[2]

    @property
    def use_bias(self):
        return hasattr(self, "bias_mat")


class ProbUnifMPS(ProbMPS):
    r"""
    Uniform MPS model using L2 probabilities for generative modeling

    Probabilities of sequential inputs are obtained via the Born rule of
    quantum mechanics, making ProbUnifMPS a "Born machine" model. Given an
    input sequence of length n, the probability assigned to the sequence
    :math:`x = x_1 x_2 \dots x_n` is :math:`P(x) = |h_n^T \omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A[x_t] + B) h_{t-1},

    with :math:`h_0 := \alpha` (for :math:`\alpha, \omega` trainable
    parameter vectors), :math:`A[i]` the i'th matrix slice of the
    third-order MPS core tensor, and :math:`B` an optional bias matrix.

    Note that calling a :class:`ProbUnifMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbUnifMPS`.

    Args:
        input_dim: Dimension of the inputs to the uMPS core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking copies of uMPS core.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
        init_method: String specifying how to initialize the MPS core tensors.
            Giving "near_eye" initializes all core slices to near the identity,
            while "normal" has all core elements be normally distributed.
            Default: ``"near_eye"``
        embed_fun: Function which embeds discrete or continous scalar values
            into vectors of dimension `input_dim`. Must be able to take in a
            tensor of any order `n` and output a tensor of order `n+1`, where
            the scalar values of the input are represented as vectors in the
            *last* axis of the output.
            Default: ``None`` (no embedding function)
        domain: Instance of the `DataDomain` class, which specifies whether
            the input data domain is continuous vs. discrete, and what range
            of values the domain takes.
            Default: ``None``
    """

    def __init__(
        self,
        input_dim: int,
        bond_dim: int,
        complex_params: bool = False,
        use_bias: bool = False,
        init_method: str = "near_eye",
        embed_fun: Optional[Callable] = None,
        domain: Optional[DataDomain] = None,
    ) -> None:
        super(ProbMPS, self).__init__()
        assert min(input_dim, bond_dim) > 0

        # Initialize core tensor and edge vectors
        assert init_method in ("near_eye", "normal")
        init_fun = near_eye_init if init_method == "near_eye" else normal_init
        core_tensors = init_fun(
            (input_dim, bond_dim, bond_dim), is_complex=complex_params
        )
        # Left and right vectors initialized to be identical, since it avoids
        # issues with exponentially small overlap
        rand_vec = torch.randn(bond_dim) / sqrt(bond_dim)
        edge_vecs = torch.stack((rand_vec, rand_vec.conj()))
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)

        # Initialize (optional) bias matrices at zero
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)

        # Set other MPS attributes
        self.complex_params = complex_params
        self.embedding = None

        # Set up embedding object if desired
        if isinstance(embed_fun, (FixedEmbedding, TrainableEmbedding)):
            self.embedding = embed_fun
            if hasattr(embed_fun, "emb_dim"):
                assert self.embedding.emb_dim == input_dim
        elif embed_fun is not None:
            assert domain is not None
            self.embedding = FixedEmbedding(embed_fun, domain)
            assert self.embedding.emb_dim == input_dim

    def forward(
        self, input_data: Tensor, slim_eval: bool = False, parallel_eval: bool = False
    ) -> Tensor:
        """
        Get the log probabilities of batch of input data

        Args:
            input_data: Sequential with shape `(batch, seq_len)`, for
                discrete inputs, or shape `(batch, seq_len, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            log_probs: Vector with shape `(batch,)` giving the natural
                logarithm of the probability of each input sequence.
        """
        batch, seq_len = input_data.shape[:2]

        # Apply embedding function if it is defined
        if self.embedding is not None:
            input_data = self.embedding(input_data)

        if slim_eval:
            if self.use_bias:
                raise ValueError("Bias matrices not supported for slim_eval")
            psi_vals, log_scales = slim_eval_fun(
                input_data, self.core_tensors, self.edge_vecs
            )
        else:
            # Contract inputs with core tensors and add bias matrices
            mat_slices = get_mat_slices(input_data, self.core_tensors)
            if self.use_bias:
                mat_slices = mat_slices + self.bias_mat[None]

            #  Contract all bond dims to get (unnormalized) prob amplitudes
            psi_vals, log_scales = contract_matseq(
                mat_slices,
                self.edge_vecs[0],
                self.edge_vecs[1],
                parallel_eval,
                log_format=True,
            )

        # Get log normalization and check for infinities
        log_norm = self.log_norm(seq_len)
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())

        # Compute unnormalized log probabilities
        log_uprobs = torch.log(torch.abs(psi_vals)) + log_scales
        assert log_uprobs.shape == (batch,)

        # Return normalized probabilities
        return 2 * log_uprobs - log_norm

    def log_norm(self, data_len) -> Tensor:
        r"""
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\log(|\psi|^2)`,
        where :math:`\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        # Account for bias matrices before calling log norm implementation
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None]
        else:
            core_tensors = self.core_tensors

        # Account for non-trivial lambda function in the embedding
        lamb_mat = None if self.embedding is None else self.embedding.lamb_mat

        return get_log_norm(
            core_tensors, self.edge_vecs, lamb_mat=lamb_mat, length=data_len
        )

    @property
    def input_dim(self):
        return self.core_tensors.shape[0]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[1]

    @property
    def use_bias(self):
        return hasattr(self, "bias_mat")
