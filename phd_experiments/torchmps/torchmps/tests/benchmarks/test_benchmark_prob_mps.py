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

"""Tests for probabilistic MPS functions"""
from functools import partial

import torch

from torchmps import ProbMPS, ProbUnifMPS
from torchmps.tests.utils_for_tests import group_name


def prob_eval_runner(
    benchmark,
    old_eval: bool = False,
    vec_input: bool = False,
    parallel: bool = False,
    uniform: bool = True,
    cmplx: bool = False,
    input_dim: int = 10,
    bond_dim: int = 10,
    seq_len: int = 100,
    batch: int = 100,
):
    # Initialize MPS model, fix evaluation style
    if uniform:
        mps_model = ProbUnifMPS(input_dim, bond_dim, cmplx, parallel)
    else:
        mps_model = ProbMPS(seq_len, input_dim, bond_dim, cmplx, parallel)
    mps_model = partial(mps_model, slim_eval=(not old_eval))

    # Create fake input data
    if vec_input:
        fake_data = torch.randn(seq_len, batch, input_dim).abs()
        fake_data = fake_data / fake_data.sum(dim=-1, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (seq_len, batch))

    # Benchmark get_mat_slices using input benchmark
    benchmark(mps_model, fake_data)


# Shorthand for old vs new evaluations
old_eval_runner = partial(prob_eval_runner, old_eval=True)
new_eval_runner = partial(prob_eval_runner, old_eval=False)


@group_name("old_eval")
def test_old_eval_base(benchmark):
    """Benchmark old probabilistic evaluation with default values"""
    old_eval_runner(benchmark)


@group_name("new_eval")
def test_new_eval_base(benchmark):
    """Benchmark new probabilistic evaluation with default values"""
    new_eval_runner(benchmark)


@group_name("old_eval")
def test_old_eval_nonuniform(benchmark):
    """Benchmark old probabilistic evaluation with non-uniform core tensors"""
    old_eval_runner(benchmark, uniform=False)


@group_name("new_eval")
def test_new_eval_nonuniform(benchmark):
    """Benchmark new probabilistic evaluation with non-uniform core tensors"""
    new_eval_runner(benchmark, uniform=False)


@group_name("old_eval")
def test_old_eval_complex(benchmark):
    """Benchmark old probabilistic evaluation with complex core tensors"""
    old_eval_runner(benchmark, cmplx=True)


@group_name("new_eval")
def test_new_eval_complex(benchmark):
    """Benchmark new probabilistic evaluation with complex core tensors"""
    new_eval_runner(benchmark, cmplx=True)


@group_name("old_eval")
def test_old_eval_large_seqlen(benchmark):
    """Benchmark old probabilistic evaluation with 10x longer sequences"""
    old_eval_runner(benchmark, seq_len=1000)


@group_name("new_eval")
def test_new_eval_large_seqlen(benchmark):
    """Benchmark new probabilistic evaluation with 10x longer sequences"""
    new_eval_runner(benchmark, seq_len=1000)


@group_name("old_eval")
def test_old_eval_vecs_in(benchmark):
    """Benchmark old probabilistic evaluation with continuous inputs"""
    old_eval_runner(benchmark, vec_input=True)


@group_name("new_eval")
def test_new_eval_vecs_in(benchmark):
    """Benchmark new probabilistic evaluation with continuous inputs"""
    new_eval_runner(benchmark, vec_input=True)


@group_name("old_eval")
def test_old_eval_large_bonddim(benchmark):
    """Benchmark old probabilistic evaluation with 10x larger bond dimensions"""
    old_eval_runner(benchmark, bond_dim=100)


@group_name("new_eval")
def test_new_eval_large_bonddim(benchmark):
    """Benchmark new probabilistic evaluation with 10x larger bond dimensions"""
    new_eval_runner(benchmark, bond_dim=100)


@group_name("old_eval")
def test_old_eval_large_inputdim(benchmark):
    """Benchmark old probabilistic evaluation with 10x larger input dimensions"""
    old_eval_runner(benchmark, input_dim=100)


@group_name("new_eval")
def test_new_eval_large_inputdim(benchmark):
    """Benchmark new probabilistic evaluation with 10x larger input dimensions"""
    new_eval_runner(benchmark, input_dim=100)


@group_name("old_eval")
def test_old_eval_extreme(benchmark):
    """Benchmark old probabilistic evaluation with larger bond dims and seq lens"""
    old_eval_runner(benchmark, bond_dim=100, seq_len=1000)


@group_name("new_eval")
def test_new_eval_extreme(benchmark):
    """Benchmark new probabilistic evaluation with larger bond dims and seq lens"""
    new_eval_runner(benchmark, bond_dim=100, seq_len=1000)
