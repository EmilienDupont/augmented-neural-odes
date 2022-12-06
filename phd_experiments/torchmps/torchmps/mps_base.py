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

"""Basic MPS functions used for uniform and non-uniform models"""
import warnings
from math import sqrt
from itertools import repeat
from typing import Union, Sequence, Optional, Tuple

import torch
from torch import Tensor

from torchmps.utils2 import (
    bundle_tensors,
    batch_broadcast,
    batch_to,
    phaseify,
    einsum,
    hermitian_trace,
    realify,
    CIndex,
)

TensorSeq = Union[Tensor, Sequence[Tensor]]


def slim_eval_fun(seq_input: Tensor, core_tensor: Tensor, bound_vecs: Tensor) -> Tensor:
    r"""
    Evaluate MPS tensor elements relative to a batch of sequential inputs.

    Args:
        seq_input: Tensor with shape `(batch, seq_len)` for discrete input
            sequences, or `(batch, seq_len, input_dim)` for vector input
            sequences.
        core_tensor: Tensor with shape `(input_dim, bond_dim, bond_dim)`
            for uniform MPS or `(seq_len, input_dim, bond_dim, bond_dim)`
            for fixed-length MPS.
        bound_vecs: Left and right boundary vectors expressed in matrix
            with shape `(2, bond_dim)`.

    Returns:
        contraction: Vector with shape `(batch,)` containing the elements
            of the MPS parameterized by `core_tensor` and `bound_vecs`,
            relative to the inputs in `seq_input`.
        log_scale: Vector with shape `(batch,)` containing (the logarithms of)
            positive-valued corrections to the scalar outputs in contraction,
            so that the actual values are `contraction * exp(log_scale)`.
    """
    # Following algo assumes batch and sequence indices are flipped
    seq_input = seq_input.transpose(0, 1)

    # Check inputs, ensure they have correct shapes
    seq_len = len(seq_input)
    batch = seq_input.shape[1]
    bond_dim = core_tensor.shape[-1]
    assert bound_vecs.ndim == 2
    assert seq_input.ndim in (2, 3)
    assert core_tensor.ndim in (3, 4)
    assert bound_vecs.shape[0] == 2
    assert bound_vecs.shape[-1] == core_tensor.shape[-1]
    uniform = core_tensor.ndim == 3
    vec_input = seq_input.ndim == 3
    if not uniform:
        assert len(core_tensor) == seq_len
    if vec_input:
        assert seq_input.shape[-1] == core_tensor.shape[-3]

    # Promote seq_input to complex dtype if core_tensor is complex
    if core_tensor.is_complex() and seq_input.is_floating_point():
        seq_input = seq_input.to(core_tensor.dtype)

    # Condition on cases of vector inputs and non-uniform MPS
    if uniform:
        all_cores = repeat(core_tensor, seq_len)
    else:
        all_cores = core_tensor
    if vec_input:
        slice_fun = lambda inps, core: einsum("bi,ide->bde", inps, core)
    else:
        slice_fun = lambda inps, core: core[inps]

    # Process input sequentially, from left to right
    log_scale = torch.zeros(batch)
    vecs = bound_vecs[0][None, None]
    for inps, core in zip(seq_input, all_cores):
        mats = slice_fun(inps, core)
        vecs = torch.matmul(vecs, mats)

        # Rescale vectors, update log_scale
        rescale = vecs.abs().sum(dim=-1, keepdim=True) / bond_dim
        log_scale = log_scale + rescale.log()[:, 0, 0]
        vecs = vecs / rescale

    # Contract with the right boundary vector, return result
    contraction = torch.matmul(vecs.squeeze(dim=1), bound_vecs[1][:, None])
    assert contraction.shape == (batch, 1)
    return contraction.squeeze(dim=1), log_scale


def contract_matseq(
    matrices: TensorSeq,
    left_vec: Optional[Tensor] = None,
    right_vec: Optional[Tensor] = None,
    parallel_eval: bool = False,
    log_format: bool = False,
) -> Tensor:
    r"""
    Matrix-multiply sequence of matrices with optional boundary vectors

    The output is a single matrix, or a vector/scalar if one/both boundary
    vectors are given. In the latter case, the first vector is treated as
    a row vector and the last as a column vector and put at the beginning
    or end of the sequence of matrices to reduce to a vector/scalar.

    Parallel matrix-matrix multiplications can be used to make the
    computation more GPU-friendly, at the cost of a larger overall compute
    cost. By default, this method is only used when an output matrix is
    desired, but can be forced by setting parallel_eval to True.

    When matrices or boundary vectors contain additional batch indices
    (assumed to be left-most indices), then batch matrix multiplication is
    carried out over all batch indices, which are broadcast together.
    Shapes described below neglect these additional batch indices, which will
    be possessed by all outputs whenever they are present in input `matrices`.

    Args:
        matrices: Single tensor of shape `(L, D, D)`, or sequence of
            matrices with compatible shapes :math:`(D_i, D_{i+1})`, for
            :math:`i = 0, 1, \ldots, L`.
        left_vec: Left boundary vector with shape `(D_0,)`, or None if no
            left boundary is present.
        right_vec: Left boundary vector with shape `(D_L,)`, or None if no
            right boundary is present.
        parallel_eval: Whether or not to force parallel evaluation in
            matrix contraction, which requires all input matrices to have
            same shape.
            Default: ``False``
        log_format: Whether or not to return the output as two objects, a
            contraction output and a logarithm scale correction for each
            product of matrices in the original input.
            Default: ``False``

    Returns:
        contraction: Single scalar, vector, or matrix, equal to the
            sequential contraction of the input matrices with (resp.)
            two, one, or zero boundary vectors.
        log_scale: Real scalar corrections to the magnitude of `contraction`,
            so that the real output is `contraction * exp(log_scale)`. Only
            present when log_format is True.
    """
    # Count number of boundary vectors
    bnd_vecs = [left_vec, right_vec]
    real_vec = [v is not None for v in bnd_vecs]
    num_vecs = sum(real_vec)
    assert all(v is None or isinstance(v, Tensor) for v in bnd_vecs)
    assert num_vecs <= 2

    # Convert matrices to single batch tensor, provided all shapes agree
    same_shape = isinstance(matrices, Tensor)
    if not same_shape:
        matrices = bundle_tensors(matrices, dim=-3)
        same_shape = isinstance(matrices, Tensor)
    num_mats = matrices.shape[-3] if same_shape else len(matrices)

    # Decide whether to use parallel evaluation algorithm
    use_parallel = same_shape and (parallel_eval or num_vecs == 0)

    # Broadcast batch dimensions of matrices and boundary vectors
    if num_vecs == 0 and not same_shape:
        matrices = batch_broadcast(matrices, (2,) * num_mats)
    elif num_vecs == 1:
        v_ind = real_vec.index(True)
        vec = bnd_vecs[v_ind]
        if same_shape:
            vec, matrices = batch_broadcast((vec, matrices), (1, 3))
        else:
            outs = batch_broadcast((vec,) + tuple(matrices), (1,) + (2,) * num_mats)
            vec, matrices = outs[0], outs[1:]
        bnd_vecs[v_ind] = vec
    elif num_vecs == 2 and same_shape:
        outs = batch_broadcast(bnd_vecs + [matrices], (1, 1, 3))
        bnd_vecs, matrices = outs[:2], outs[2]
    elif num_vecs == 2 and not same_shape:
        outs = batch_broadcast(bnd_vecs + list(matrices), (1, 1) + (2,) * num_mats)
        bnd_vecs, matrices = outs[:2], outs[2:]

    if use_parallel:
        # Reduce product of all matrices in parallel
        product, log_scale = mat_reduce_par(matrices)

        # Contract with boundary vectors, using intermediate dummy axes
        if real_vec[0]:
            product = torch.matmul(bnd_vecs[0][..., None, :], product)
        if real_vec[1]:
            product = torch.matmul(product, bnd_vecs[1][..., None])

    else:
        if num_vecs == 0 and len(matrices) == 0:
            raise ValueError(
                "Must input at least one matrix or boundary vector to contract_matseq"
            )

        # Prepend/append boundary vectors, augmented with dummy dimensions
        if same_shape:
            matrices = [matrices[..., i, :, :] for i in range(num_mats)]
        else:
            matrices = list(matrices)
        if real_vec[0]:
            matrices = [bnd_vecs[0][..., None, :]] + matrices
        if real_vec[1]:
            matrices.append(bnd_vecs[1][..., None])

        # Compute product sequentially and strip away dummy dimensions
        product, log_scale = mat_reduce_seq(matrices)

    # Strip away dummy indices
    if real_vec[0]:
        product.squeeze_(-2)
        log_scale.squeeze_(-2)
    if real_vec[1]:
        product.squeeze_(-1)
        log_scale.squeeze_(-1)

    if log_format:
        return product, log_scale
    else:
        return product * torch.exp(log_scale)


def mat_reduce_par(matrices: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Contract sequence of square matrices with parallel mat-mat multiplies

    Args:
        matrices: Sequence of matrices to multiply, specified as a single
            tensor with shape `(batch, seq_len, bond_dim, bond_dim)`.

    Returns:
        prod_mat: Product of input matrices.
        log_scale: Vector with shape `(batch, 1, 1)` containing the logarithms
            of positive-valued corrections to the matrices in `prod_mat`, so
            that the actual values are `prod_mat * exp(log_scale)`.
    """
    assert matrices.ndim >= 3
    s_dim = -3  # Dimension which has spatial arrangement of matrices
    n_mats = matrices.shape[s_dim]

    # Initialize the register for the log scale factors
    log_scale = torch.zeros(matrices.shape[:-3])[..., None, None]

    # In case of empty collection of matrices, return the identity
    if n_mats == 0:
        eye = torch.eye(matrices.shape[-1], dtype=matrices.dtype)
        matrices, _ = batch_broadcast((eye, matrices), (2, 3))
        return matrices, log_scale
    elif n_mats == 1:
        return matrices.squeeze(dim=s_dim), log_scale

    # Iteratively multiply pairs of matrices until there is only one left
    assert matrices.shape[-2] == matrices.shape[-1]
    bond_dim = matrices.shape[-1]
    while n_mats > 1:
        half_n = n_mats // 2
        floor_n = half_n * 2

        # Split matrices up into even and odd numbers (maybe w/ leftover)
        even_mats = matrices[..., 0:floor_n:2, :, :]
        odd_mats = matrices[..., 1:floor_n:2, :, :]
        leftover = matrices[..., floor_n:, :, :]

        # Batch multiply everything, append remainder
        matrices = even_mats @ odd_mats
        matrices = torch.cat((matrices, leftover), dim=s_dim)
        n_mats = matrices.shape[s_dim]

        # Rescale matrices and update log_scale
        rescales = matrices.abs().sum(dim=(-2, -1), keepdim=True) / bond_dim
        log_scale = log_scale + rescales.log().sum(dim=-3)
        matrices = matrices / rescales

    return matrices.squeeze(dim=s_dim), log_scale


def mat_reduce_seq(matrices: Sequence[Tensor]) -> Tensor:
    """
    Multiply sequence of matrices sequentially, from left to right

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices.
        log_scale: Logarithms of positive-valued corrections to the matrices
        in `prod_mat`, so that actual values are `prod_mat * exp(log_scale)`.
    """
    # Multiplication from left to right, so flip matrices if it's cheaper
    # to multiply from right to left
    r2l = matrices[-1].size(-1) < matrices[0].size(-2)
    if r2l:
        matrices = tuple(m.transpose(-2, -1) for m in matrices[::-1])

    # Multiply all matrices sequentially, from left to right
    product, matrices = matrices[0], matrices[1:]
    log_scale = torch.zeros(product.shape[:-2])[..., None, None]
    for mat in matrices:
        product = torch.matmul(product, mat)

        # Rescale and update the log scale factor
        target_norm = sqrt(product.shape[-2] * product.shape[-1])
        rescale = product.abs().sum(dim=(-2, -1), keepdim=True) / target_norm
        log_scale = log_scale + torch.log(rescale)
        product = product / rescale

    # Revert to original form before returning
    product = product.transpose(-2, -1) if r2l else product
    return product, log_scale


def get_mat_slices(seq_input: Tensor, core_tensor: Tensor) -> Tensor:
    """
    Use sequential input and core tensor to get sequence of matrix slices

    Args:
        seq_input: Tensor with shape `(batch, seq_len)` for discrete input
            sequences, or `(batch, seq_len, input_dim)` for vector input
            sequences.
        core_tensor: Tensor with shape `(input_dim, bond_dim, bond_dim)`
            for uniform MPS or `(seq_len, input_dim, bond_dim, bond_dim)`
            for fixed-length MPS.

    Returns:
        mat_slices: Tensor with shape `(batch, seq_len, bond_dim, bond_dim)`,
            containing all transition matrix core slices of `core_tensor`
            relative to data in `seq_input`.
    """
    # Set flags for which case we're dealing with
    assert seq_input.ndim in (2, 3)
    assert core_tensor.ndim in (3, 4)
    indexing = seq_input.ndim == 2
    uniform = core_tensor.ndim == 3
    input_dim, bond_dim = core_tensor.shape[-3:-1]
    seq_len = seq_input.shape[1]

    # TODO: Deal with packing if/when we have variable-len sequences,
    #       likely using pad_mat_slices function

    # Promote seq_input to complex dtype if core_tensor is complex
    if core_tensor.is_complex() and seq_input.is_floating_point():
        seq_input = seq_input.to(core_tensor.dtype)

    # Perform the indexing/contraction on a case-by-case basis
    if indexing and uniform:
        mat_slices = core_tensor[seq_input]
    elif indexing and not uniform:
        # Indexing equivalent of the einsum operation at bottom
        mat_slices = CIndex(core_tensor)[torch.arange(seq_len)[None], seq_input]
        # mat_slices = core_tensor[torch.arange(seq_len)[None], seq_input]
    elif not indexing and uniform:
        mat_slices = einsum("bti,ide->btde", seq_input, core_tensor)
    elif not indexing and not uniform:
        mat_slices = einsum("bti,tide->btde", seq_input, core_tensor)

    return mat_slices


def get_log_norm(
    core_tensor: Tensor,
    boundary_vecs: Tensor,
    length: Optional[int] = None,
    lamb_mat: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Compute the log (squared) L2 norm of tensor described by MPS model

    Uses iterated tensor contraction to compute :math:`\log(|\psi|^2)`,
    where :math:`\psi` is the n'th order tensor arising from contracting
    all MPS core tensors and boundary vectors together. In the Born machine
    paradigm this is equivalently :math:`\log(Z)`, with :math:`Z` the
    normalization constant for the probability.

    Can be used to compute fixed-len log norms as well as arbitrary-len
    log norms, for the case of uniform MPS. The latter case is the log of
    the sum of all len-n squared norms, for :math:`n = 0, 1, \ldots`.
    Non-convergent core tensors

    Args:
        core_tensor: Either single core tensor of shape `(input_dim, D, D)`
            or core tensor sequence of shape `(seq_len, input_dim, D, D)`,
            for `D` the bond dimension of the MPS.
        boundary_vecs: Matrix of shape `(2, D)` whose 0th/1st columns give
            the left/right boundary vectors of the MPS.
        length: In case of single core tensor, specifies the length for
            which the log L2 norm is computed. Non-negative values give
            fixed-len log norms, while -1 gives arbitrary-len log norm.
        lamb_mat: Positive semidefinite matrix :math:`\Lambda` which is used
            in the presence of an input embedding to marginalize over all all
            values of the input domain.

    Returns:
        l_norm: Scalar value giving the log squared L2 norm of the
            probability amplitude tensor described by the MPS.
    """
    # Determine whether normalization is fixed-len or arbitrary-len
    uniform = core_tensor.ndim == 3
    assert core_tensor.ndim in (3, 4)
    assert boundary_vecs.shape[-1] == core_tensor.shape[-1]
    assert uniform or length in (None, core_tensor.shape[0])
    assert lamb_mat is None or lamb_mat.ndim <= 2
    assert not (uniform and length is None)
    fixed_len = not uniform or length != -1

    # Initialize all variables needed for log norm calculation
    uniform = core_tensor.ndim == 3
    left_vec, r_vec = boundary_vecs
    dens_mat = left_vec[:, None] @ left_vec[None, :].conj()
    log_norm = torch.zeros(())
    if uniform and fixed_len:
        core_tensor = (core_tensor,) * length

    # Define transfer operator function for different types of lambda matrix
    if lamb_mat is None:
        # In the following, "dm" = "density matrix", "ct" = "core tensor"
        t_op = lambda dm, ct: einsum("ilr,lp,ipq->rq", ct, dm, ct.conj())
    elif lamb_mat.ndim == 0:
        t_op = lambda dm, ct: lamb_mat * einsum("ilr,lp,ipq->rq", ct, dm, ct.conj())
    elif lamb_mat.ndim == 1:
        t_op = lambda dm, ct: einsum("i,ilr,lp,ipq->rq", lamb_mat, ct, dm, ct.conj())
    elif lamb_mat.ndim == 2:
        t_op = lambda dm, ct: einsum("ij,ilr,lp,jpq->rq", lamb_mat, ct, dm, ct.conj())

    # Function for applying rightward transfer operator to density mat
    def transfer_op(core_t, d_mat, l_norm, l_sum=None):
        # Update density operator, rescale to have unit trace
        d_mat = t_op(d_mat, core_t)
        trace = hermitian_trace(d_mat)
        l_norm = l_norm + torch.log(trace)
        d_mat = d_mat / trace

        if fixed_len:
            return d_mat, l_norm
        else:
            # Get new contribution to log L2 norm for variable-len case
            this_norm = einsum("r,rq,q->", r_vec, d_mat, r_vec.conj())
            new_lnorm = torch.log(this_norm) + l_norm
            l_sum = torch.logsumexp(torch.stack((l_sum, new_lnorm)))
            return d_mat, l_norm, l_sum

    if fixed_len:
        # Apply transfer operator `length` times, then add final correction
        for core in core_tensor:
            dens_mat, log_norm = transfer_op(core, dens_mat, log_norm)

        correction = einsum("r,rq,q->", r_vec, dens_mat, r_vec.conj())
        return log_norm + torch.log(realify(correction))
    else:
        raise NotImplementedError

        # TODO: Deal with the more intricate arbitrary-len calculation
        # log_sum = torch.tensor(-float("inf"))
        # Use transfer_op with log_sum argument, iterate until convergence


# TODO: Complete this after variable-len data format is better figured out
def pad_mat_slices():
    pass


def near_eye_init(
    shape: tuple, is_complex: bool = False, noise: float = 1e-3
) -> Tensor:
    """
    Initialize an MPS core tensor with all slices close to identity matrix

    Args:
        shape: Shape of the core tensor being initialized.
        is_complex: Whether to initialize a complex core tensor.
            Default: False
        noise: Normalized noise value setting stdev around identity matrix.
            Default: 1e-3

    Returns:
        core_tensor: Randomly initialized near-identity core tensor.
    """
    # Check shape and do something if core slices are non-square
    assert len(shape) >= 3
    if shape[-2] != shape[-1]:
        if torch.prod(torch.tensor(shape[:-3])) != 1:
            raise ValueError(
                "Batch core tensor with non-square matrix slices "
                "requested, pretty sure this isn't what you wanted"
            )
        else:
            warnings.warn(
                "Core tensor with non-square matrix slices "
                "requested, is this really what you wanted?"
            )

    # Initialize core using size-adjusted value of noise
    eye_core = batch_to(torch.eye(*shape[-2:]), shape[:-2], 2)
    noise = noise / torch.sqrt(torch.prod(torch.tensor(shape[-2:])).float())
    delta = noise * torch.randn(shape)
    if is_complex:
        delta = phaseify(delta)
    return eye_core + delta


def normal_init(shape: tuple, is_complex: bool = False, rel_std: float = 1.0) -> Tensor:
    """
    Initialize an MPS core tensor with all slices normally distributed

    Args:
        shape: Shape of the core tensor being initialized.
        is_complex: Whether to initialize a complex core tensor.
            Default: False
        rel_std: Relative standard deviation of entries of matrix slices,
            scaled by a factor of the bond dimension of the MPS.
            Default: 1.0

    Returns:
        core_tensor: Normally distributed near-identity core tensor.
    """
    # Check shape and do something if core slices are non-square
    assert len(shape) >= 3
    if shape[-2] != shape[-1]:
        if torch.prod(torch.tensor(shape[:-3])) != 1:
            raise ValueError(
                "Batch core tensor with non-square matrix slices "
                "requested, pretty sure this isn't what you wanted"
            )
        else:
            warnings.warn(
                "Core tensor with non-square matrix slices "
                "requested, is this really what you wanted?"
            )

    # Initialize core using size-adjusted value of variance
    std = 1 / shape[-1]
    core_tensor = std * torch.randn(shape)
    if is_complex:
        core_tensor = phaseify(core_tensor)
    return core_tensor
