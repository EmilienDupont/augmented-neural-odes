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

"""Utility functions that are only used for tests"""
import pytest

import numpy as np
import torch

Tensor = torch.Tensor


def complete_binary_dataset(str_len: int) -> Tensor:
    """
    Generate dataset with all binary strings of given length
    """
    input_dim = 2
    samp_size = input_dim ** str_len
    all_data = np.zeros((samp_size, str_len), dtype="int32")
    for i in range(samp_size):
        bin_str = bin(i)[2:]
        bin_str = "0" * (str_len - len(bin_str)) + bin_str
        for n, b in enumerate(bin_str):
            all_data[i, n] = int(b)
    return torch.tensor(all_data).long()


def allcloseish(arr1: Tensor, arr2: Tensor, tol=1e-4) -> bool:
    """
    Same as `torch.allclose`, but less nit-picky
    """
    if not isinstance(arr1, torch.Tensor):
        arr1 = torch.tensor(arr1)
    if not isinstance(arr2, torch.Tensor):
        arr2 = torch.tensor(arr2)
    return torch.allclose(arr1, arr2, rtol=tol, atol=tol)


def group_name(name: str):
    """Convenience wrapper for setting pytest benchmark group names"""
    return pytest.mark.benchmark(group=name)
