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

"""Tests for MPS base functions"""
import pytest
from math import sqrt
from functools import partial

import torch
from hypothesis import given, settings, strategies as st

from torchmps.mps_base import (
    contract_matseq,
    get_mat_slices,
    near_eye_init,
    slim_eval_fun,
)
from torchmps.utils2 import batch_broadcast, batch_to
from torchmps.tests.utils_for_tests import allcloseish

bool_st = st.booleans
seq_len_st = partial(st.integers, 1, 1000)
bond_dim_st = partial(st.integers, 1, 20)
input_dim_st = partial(st.integers, 1, 10)


def batch_shape_st(s_len):
    return st.lists(st.integers(1, 10), min_size=0, max_size=s_len)


def naive_contraction(mats, lvec, rvec):
    """Handle conditional contraction with boundary vectors"""
    # For empty batch of matrices, replace with single identity matrix
    use_lvec, use_rvec = lvec is not None, rvec is not None
    if isinstance(mats, torch.Tensor) and mats.shape[-3] == 0:
        assert isinstance(mats, torch.Tensor)
        assert mats.shape[-2] == mats.shape[-1]
        eye = torch.eye(mats.shape[-1])
        mats = [batch_to(eye, mats.shape[:-3], 2)]

    # Convert mats to list, add phony dims to vecs, broadcast all batch dims
    if isinstance(mats, torch.Tensor):
        mats = [mats[..., i, :, :] for i in range(mats.shape[-3])]
    else:
        assert hasattr(mats, "__len__")
        mats = list(mats)
    if use_lvec:
        lvec = lvec[..., None, :]
        out = batch_broadcast([lvec] + mats, (2,) * (len(mats) + 1))
        lvec, mats = out[0], list(out[1:])
    if use_rvec:
        rvec = rvec[..., :, None]
        out = batch_broadcast([rvec] + mats, (2,) * (len(mats) + 1))
        rvec, mats = out[0], out[1:]

    # Matrix/vector multiplication which respects batch dimensions
    if use_rvec:
        for mat in mats[::-1]:
            rvec = torch.matmul(mat, rvec)
        rvec = rvec[..., 0]
        if not use_lvec:
            return rvec
        else:
            return torch.sum(lvec[..., 0, :] * rvec, dim=-1)
    elif use_lvec:
        for mat in mats:
            lvec = torch.matmul(lvec, mat)
        lvec = lvec[..., 0, :]
        return lvec
    else:
        pmat = mats[0]
        for mat in mats[1:]:
            pmat = torch.matmul(pmat, mat)
        return pmat


@given(
    seq_len_st(),
    bond_dim_st(),
    input_dim_st(),
    input_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_get_mat_slices_shape(
    seq_len: int,
    bond_dim: int,
    input_dim: int,
    batch: int,
    vec_input: bool,
    is_complex: bool,
    uniform: bool,
):
    """
    Check that get_mat_slices gives correct shapes
    """
    if uniform:
        core_shape = (input_dim, bond_dim, bond_dim)
    else:
        core_shape = (seq_len, input_dim, bond_dim, bond_dim)
    core_tensor = near_eye_init(core_shape, is_complex)
    assert core_tensor.is_complex() == is_complex

    if vec_input:
        fake_data = torch.randn(batch, seq_len, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (batch, seq_len))

    # Run get_mat_slices and verify that the output has expected shape
    output = get_mat_slices(fake_data, core_tensor)
    assert output.shape == (batch, seq_len, bond_dim, bond_dim)


@settings(deadline=None)
@given(
    seq_len_st(),
    bond_dim_st(),
    input_dim_st(),
    input_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_composite_init_mat_slice_contraction(
    seq_len: int,
    bond_dim: int,
    input_dim: int,
    batch: int,
    vec_input: bool,
    is_complex: bool,
    uniform: bool,
):
    """
    Verify that initializing identity core, getting matrix slices, and then
    contracting the slices gives identity matrices
    """
    if uniform:
        core_shape = (input_dim, bond_dim, bond_dim)
    else:
        core_shape = (seq_len, input_dim, bond_dim, bond_dim)
    core_tensor = near_eye_init(core_shape, is_complex, noise=0)
    assert core_tensor.is_complex() == is_complex

    if vec_input:
        fake_data = torch.randn(batch, seq_len, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (batch, seq_len))

    # Get matrix slices, then contract them all together
    mat_slices = get_mat_slices(fake_data, core_tensor)

    prod_mats = contract_matseq(mat_slices)

    # Verify that all contracted matrix slices are identities
    target_prods = torch.eye(bond_dim)
    assert torch.allclose(prod_mats.abs(), target_prods, atol=1e-4, rtol=1e-4)

    # Do the same thing for slim_eval_fun, but with boundary vectors
    ref_vec = torch.randn(bond_dim).to(core_tensor.dtype)
    ref_vals = ref_vec.norm() ** 2
    bound_vecs = torch.stack((ref_vec, ref_vec))
    prod_vals, log_scales = slim_eval_fun(fake_data, core_tensor, bound_vecs)
    prod_vals *= log_scales.exp()
    assert torch.allclose(prod_vals.abs(), ref_vals, atol=1e-4, rtol=1e-4)


@settings(deadline=None)
@given(
    batch_shape_st(4),
    bond_dim_st(),
    st.integers(1, 10),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_contract_matseq_identity_batches(
    batch_shape, bond_dim, seq_len, use_lvec, use_rvec, parallel_eval
):
    """
    Multipy random multiples of the identity matrix w/ variable batch size
    """
    # Generate identity matrices and boundary vectors
    shape = tuple(batch_shape) + (seq_len, bond_dim, bond_dim)
    eye_mats = near_eye_init(shape, noise=0)
    eye_mats2 = [eye_mats[..., i, :, :] for i in range(seq_len)]
    left_vec, right_vec = torch.randn(2, bond_dim)
    lvec = left_vec if use_lvec else None
    rvec = right_vec if use_rvec else None

    # Contract with the naive algorithm, compare to contract_matseq output
    naive_result = naive_contraction(eye_mats, lvec, rvec)
    lib_result = contract_matseq(eye_mats, lvec, rvec, parallel_eval)
    lib_result2 = contract_matseq(eye_mats2, lvec, rvec, parallel_eval)

    # Both ways of calling contract_matseq should agree
    assert torch.equal(lib_result, lib_result2)
    assert torch.allclose(lib_result, naive_result)


@given(
    st.lists(bond_dim_st(), min_size=1, max_size=10),
    batch_shape_st(3),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_contract_matseq_random_inhom_bonddim(
    bonddim_list, vec_batch, use_lvec, use_rvec, use_tuple, parallel_eval
):
    """
    Multiply random matrices with inhom bond dimensions, boundary vecs have batch dims
    """
    # Generate random matrices and batch boundary vectors
    num_bd, bd_lst = len(bonddim_list), bonddim_list
    rescales = [sqrt(bd_lst[i] * bd_lst[i + 1]) for i in range(num_bd - 1)]
    matrices = [
        torch.randn(bd_lst[i], bd_lst[i + 1]) / r for i, r in enumerate(rescales)
    ]
    if use_tuple:
        matrices = tuple(matrices)
    left_vec = torch.randn(*(vec_batch + [bd_lst[0]]))
    right_vec = torch.randn(*(vec_batch + [bd_lst[-1]]))
    lvec = left_vec if use_lvec else None
    rvec = right_vec if use_rvec else None

    # If no matrices and no boundary vecs are given, contract_matseq
    # should raise an error
    if len(bonddim_list) == 1 and not (use_lvec or use_rvec):
        with pytest.raises(ValueError):
            contract_matseq(matrices, lvec, rvec, parallel_eval)
        return

    # Contract with the naive algorithm, compare to contract_matseq output
    naive_result = naive_contraction(matrices, lvec, rvec)
    lib_result = contract_matseq(matrices, lvec, rvec, parallel_eval)

    # Numerical error is sometimes greater than tolerance of allclose
    if not torch.allclose(lib_result, naive_result):
        assert (lib_result - naive_result).norm() < 1e-5


@given(st.booleans())
def test_contract_matseq_empty(parallel_eval):
    """Verify that no boundary vectors and empty sequence raises error"""
    with pytest.raises(ValueError):
        contract_matseq((), None, None, parallel_eval)


@settings(deadline=None)
@given(st.integers(1, 5000), st.booleans())
def test_contract_functions_stable(seq_len, parallel_eval):
    """
    Ensure that the product of long sequence of rank-1 matrices gives the right
    log value
    """
    # Get the matrix to be multiplied and the correct scale of all matrices
    bound_vecs = torch.randn(2, 2).abs()
    rvec, lvec = bound_vecs
    mat = lvec[:, None] @ rvec[None]
    log_scalar = mat.trace().log()
    correct_log_val = seq_len * log_scalar
    matrices = torch.stack((mat,) * seq_len)

    # Compute the product of the matrices in several ways
    prod_mat1, log_scale1 = contract_matseq(
        matrices, parallel_eval=parallel_eval, log_format=True
    )
    log_val1 = log_scale1[0, 0] + prod_mat1.trace().log()
    prod_val2, log_scale2 = contract_matseq(
        matrices, rvec, lvec, parallel_eval=parallel_eval, log_format=True
    )
    log_val2 = log_scale2 + prod_val2.log()
    fake_input, fake_core = torch.zeros(1, seq_len), mat[None]
    prod_val3, log_scale3 = slim_eval_fun(fake_input.long(), fake_core, bound_vecs)
    log_val3 = log_scale3 + prod_val3.log()

    assert allcloseish(log_val1, correct_log_val)
    assert allcloseish(log_val2, correct_log_val + log_scalar)
    assert allcloseish(log_val3, correct_log_val + log_scalar)
