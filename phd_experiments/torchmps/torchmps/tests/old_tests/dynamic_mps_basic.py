#!/usr/bin/env python3
import torch
import sys

sys.path.append("..")
from torchmps import MPS

batch_size = 3
input_size = 7
output_dim = 4
bond_dim = 5
merge_threshold = 3 * batch_size

input_data = torch.randn([batch_size, input_size])

# For both open and periodic boundary conditions, place the label site in
# different locations and check that the basic behavior is correct
for bc in [False, True]:
    for num_params, label_site in [
        (8, None),
        (5, 0),
        (6, 1),
        (5, input_size),
        (6, input_size - 1),
    ]:
        mps_module = MPS(
            input_size,
            output_dim,
            bond_dim,
            periodic_bc=bc,
            label_site=label_site,
            adaptive_mode=True,
            merge_threshold=merge_threshold,
        )
        if not len(list(mps_module.parameters())) == num_params:
            print(len(list(mps_module.parameters())))
            print(num_params)
        assert len(list(mps_module.parameters())) == num_params
        assert mps_module.linear_region.offset == 0

        for _ in range(6):
            output = mps_module(input_data)
            assert list(output.shape) == [batch_size, output_dim]

        # At this point we should have flipped our offset from 0 to 1, but are
        # on the threshold so that the next call will flip offset back to 0
        assert len(list(mps_module.parameters())) == num_params
        assert mps_module.linear_region.offset == 1

        output = mps_module(input_data)
        assert list(output.shape) == [batch_size, output_dim]
        assert mps_module.linear_region.offset == 0
        assert len(list(mps_module.parameters())) == num_params
