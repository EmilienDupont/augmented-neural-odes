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
# import pytest

import torch

from torchmps.mps_base import (
    contract_matseq,
    get_mat_slices,
    near_eye_init,
    slim_eval_fun,
)
from torchmps.tests.test_mps_base import naive_contraction
from torchmps.tests.utils_for_tests import group_name


# GET_MAT_SLICES BENCHMARKS


def get_mat_slices_runner(
    benchmark,
    vec_input: bool = False,
    uniform: bool = True,
    input_dim: int = 10,
    bond_dim: int = 10,
    seq_len: int = 100,
    batch: int = 100,
):
    # Create fake core tensor and input data
    if uniform:
        core_tensor = near_eye_init((input_dim, bond_dim, bond_dim))
    else:
        core_tensor = near_eye_init((seq_len, input_dim, bond_dim, bond_dim))
    if vec_input:
        fake_data = torch.randn(seq_len, batch, input_dim).abs()
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch))

    # Benchmark get_mat_slices using input benchmark
    benchmark(get_mat_slices, fake_data, core_tensor)


@group_name("get_mat_slices")
def test_get_mat_slices_base(benchmark):
    """Benchmark get_mat_slices with default values"""
    get_mat_slices_runner(benchmark)


@group_name("get_mat_slices")
def test_get_mat_slices_nonuniform(benchmark):
    """Benchmark get_mat_slices with non-uniform core tensors"""
    get_mat_slices_runner(benchmark, uniform=False)


@group_name("get_mat_slices")
def test_get_mat_slices_large_seqlen(benchmark):
    """Benchmark get_mat_slices with 10x longer sequences"""
    get_mat_slices_runner(benchmark, seq_len=1000)


@group_name("get_mat_slices")
def test_get_mat_slices_vecs_in(benchmark):
    """Benchmark get_mat_slices with continuous inputs"""
    get_mat_slices_runner(benchmark, vec_input=True)


@group_name("get_mat_slices")
def test_get_mat_slices_large_bonddim(benchmark):
    """Benchmark get_mat_slices with 10x larger bond dimensions"""
    get_mat_slices_runner(benchmark, bond_dim=100)


@group_name("get_mat_slices")
def test_get_mat_slices_large_inputdim(benchmark):
    """Benchmark get_mat_slices with 10x larger input dimensions"""
    get_mat_slices_runner(benchmark, input_dim=100)


# CONTRACT_MATSEQ BENCHMARKS


def contract_matseq_runner(
    benchmark,
    boundaries: bool = True,
    parallel: bool = False,
    naive: bool = False,
    seq_len: int = 100,
    bond_dim: int = 10,
    batch: int = 100,
):
    # Create fake matrix slices and boundary vectors
    mat_slices = near_eye_init((seq_len, batch, bond_dim, bond_dim))
    if boundaries:
        left_vec, right_vec = torch.randn(2, bond_dim)
    else:
        left_vec, right_vec = None, None

    # Benchmark contract_matseq or naive_contraction using input benchmark
    if naive:
        benchmark(naive_contraction, mat_slices, left_vec, right_vec)
    else:
        benchmark(contract_matseq, mat_slices, left_vec, right_vec, parallel)


@group_name("contract_matseq")
def test_contract_matseq_base(benchmark):
    """Benchmark contract_matseq with default values"""
    contract_matseq_runner(benchmark)


@group_name("contract_matseq")
def test_contract_matseq_naive(benchmark):
    """Benchmark contract_matseq with naive contractions function"""
    contract_matseq_runner(benchmark, naive=True)


@group_name("contract_matseq")
def test_contract_matseq_parallel(benchmark):
    """Benchmark contract_matseq with parallel evaluation"""
    contract_matseq_runner(benchmark, parallel=True)


@group_name("contract_matseq")
def test_contract_matseq_mats_only(benchmark):
    """Benchmark contract_matseq with no boundaries"""
    contract_matseq_runner(benchmark, boundaries=False)


@group_name("contract_matseq")
def test_contract_matseq_large_seqlen(benchmark):
    """Benchmark contract_matseq with 10x longer sequences"""
    contract_matseq_runner(benchmark, seq_len=1000)


@group_name("contract_matseq")
def test_contract_matseq_large_bonddim(benchmark):
    """Benchmark contract_matseq with 10x larger bond dimensions"""
    contract_matseq_runner(benchmark, bond_dim=100)


# DEFAULT_EVAL BENCHMARKS


def slim_eval_runner(
    benchmark,
    vec_input: bool = False,
    uniform: bool = True,
    input_dim: int = 10,
    bond_dim: int = 10,
    seq_len: int = 100,
    batch: int = 100,
):
    # Create fake input data, core tensor, and boundary vectors
    if uniform:
        core_tensor = near_eye_init((input_dim, bond_dim, bond_dim))
    else:
        core_tensor = near_eye_init((seq_len, input_dim, bond_dim, bond_dim))
    if vec_input:
        fake_data = torch.randn(seq_len, batch, input_dim).abs()
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch))
    bound_vecs = torch.randn(2, bond_dim)

    # Benchmark slim_eval_fun using input benchmark
    benchmark(slim_eval_fun, fake_data, core_tensor, bound_vecs)


@group_name("slim_eval")
def test_slim_eval_base(benchmark):
    """Benchmark slim_eval_fun with default values"""
    slim_eval_runner(benchmark)


@group_name("slim_eval")
def test_slim_eval_nonuniform(benchmark):
    """Benchmark slim_eval_fun with non-uniform core tensors"""
    slim_eval_runner(benchmark, uniform=False)


@group_name("slim_eval")
def test_slim_eval_large_seqlen(benchmark):
    """Benchmark slim_eval_fun with 10x longer sequences"""
    slim_eval_runner(benchmark, seq_len=1000)


@group_name("slim_eval")
def test_slim_eval_vecs_in(benchmark):
    """Benchmark slim_eval_fun with continuous inputs"""
    slim_eval_runner(benchmark, vec_input=True)


@group_name("slim_eval")
def test_slim_eval_large_bonddim(benchmark):
    """Benchmark slim_eval_fun with 10x larger bond dimensions"""
    slim_eval_runner(benchmark, bond_dim=100)


@group_name("slim_eval")
def test_slim_eval_large_inputdim(benchmark):
    """Benchmark slim_eval_fun with 10x larger input dimensions"""
    slim_eval_runner(benchmark, input_dim=100)
