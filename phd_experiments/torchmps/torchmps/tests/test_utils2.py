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

"""Tests for utility functions"""
import torch
from hypothesis import given, strategies as st

from torchmps.utils2 import batch_broadcast


def batch_shape_st(s_len):
    return st.lists(st.integers(1, 10), min_size=0, max_size=s_len)


def nbatch_shape_st(s_len):
    return st.lists(st.integers(0, 10), min_size=0, max_size=s_len)


def test_batch_broadcast_good_shapes():
    """Run batch broadcast with shapes that should work"""
    t_shape = (5, 3, 2, 2, 2)
    m_shape = (1, 3, 3, 4)
    v_shape = (3,)
    s_shape = (2, 1, 1)

    tensors = [torch.ones(*shp) for shp in (t_shape, m_shape, v_shape, s_shape)]
    non_batch = (3, 2, 1, 0)
    out_tensors = batch_broadcast(tensors, non_batch)
    shapes = [tuple(t.shape) for t in out_tensors]

    assert shapes == [(2, 5, 3, 2, 2, 2), (2, 5, 3, 3, 4), (2, 5, 3, 3), (2, 5, 3)]


@given(
    batch_shape_st(4),
    nbatch_shape_st(3),
    nbatch_shape_st(3),
    st.integers(0, 4),
    st.booleans(),
)
def test_batch_broadcast_var_batch(
    batch_shape, nonbatch1, nonbatch2, short_len, first_short
):
    """Run batch broadcast with two tensors of variable sizes"""
    # Generate shapes for the two tensors, with smaller batch on the second
    num_batch = len(batch_shape)
    short_blen = min(num_batch, short_len)
    shape1 = batch_shape + nonbatch1
    shape2 = batch_shape + nonbatch2
    out_batch = batch_shape

    # Randomly make some batch indices singletons
    triv_vec1 = torch.randint(2, (num_batch,))
    triv_vec2 = torch.randint(2, (num_batch,))
    triv_vec2[: num_batch - short_blen] = 1
    for i in range(num_batch):
        if triv_vec1[i]:
            shape1[i] = 1
        if triv_vec2[i]:
            shape2[i] = 1
        if triv_vec1[i] and triv_vec2[i]:
            out_batch[i] = 1

    # Produce actual tensors and call batch_broadcast
    shape2 = shape2[num_batch - short_blen :]
    tensor1 = torch.empty(tuple(shape1))
    tensor2 = torch.empty(tuple(shape2))
    num_nb1, num_nb2 = len(nonbatch1), len(nonbatch2)
    if first_short:
        tensor1, tensor2 = tensor2, tensor1
        num_nb1, num_nb2 = num_nb2, num_nb1
    tensor1, tensor2 = batch_broadcast((tensor1, tensor2), (num_nb1, num_nb2))
    if first_short:
        tensor1, tensor2 = tensor2, tensor1
        num_nb1, num_nb2 = num_nb2, num_nb1

    assert tuple(tensor1.shape) == tuple(out_batch + nonbatch1)
    assert tuple(tensor2.shape) == tuple(out_batch + nonbatch2)
