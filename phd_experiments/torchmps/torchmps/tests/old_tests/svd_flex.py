#!/usr/bin/env python3
import torch
import sys

sys.path.append("..")
from utils import svd_flex

shapes = [[2, 4], [7, 2, 4], [3, 2, 5], [3, 1, 2], [2, 3, 5, 7]]

svd_strs = ["lr->lu,ur", "olr->olu,ur", "lri->lu,uri", "lri->lu,iur", "olri->olru,ui"]

out_shapes = [
    ([2, -1], [-1, 4]),
    ([7, 2, -1], [-1, 4]),
    ([3, -1], [-1, 2, 5]),
    ([3, -1], [2, -1, 1]),
    ([2, 3, 5, -1], [-1, 7]),
]

max_D = 10

tensors = [torch.randn(shape) for shape in shapes]

for i in range(len(svd_strs)):
    tensor = tensors[i]
    svd_str = svd_strs[i]
    left_shape, right_shape = out_shapes[i]

    # Run svd_flex with no specified max_D
    left_out, right_out, bond_dim = svd_flex(tensor, svd_str)

    assert len(left_shape) == len(left_out.shape)
    assert len(right_shape) == len(right_out.shape)
    for i in range(len(left_shape)):
        assert left_shape[i] == -1 or left_shape[i] == left_out.size(i)
    for i in range(len(right_shape)):
        assert right_shape[i] == -1 or right_shape[i] == right_out.size(i)

    # Run svd_flex with max_D specified and sv_right false and check again
    left_out, right_out, bond_dim = svd_flex(
        tensor, svd_str, max_D=max_D, sv_right=False
    )

    assert len(left_shape) == len(left_out.shape)
    assert len(right_shape) == len(right_out.shape)
    for i in range(len(left_shape)):
        assert left_shape[i] == -1 or left_shape[i] == left_out.size(i)
    for i in range(len(right_shape)):
        assert right_shape[i] == -1 or right_shape[i] == right_out.size(i)
