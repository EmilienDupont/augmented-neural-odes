#!/usr/bin/env python3
import torch
import sys

sys.path.append("..")
from torchmps import MPS

batch_size = 11
input_size = 21
output_dim = 4
bond_dim = 5

input_data = torch.randn([batch_size, input_size])

# For both open and periodic boundary conditions, place the label site in
# different locations and check that the basic behavior is correct
for bc in [False, True]:
    for num_params, label_site in [(3, None), (2, 0), (2, input_size)]:
        # MPS(input_size, output_dim, bond_dim, d=2, label_site=None,
        #     periodic_bc=False, parallel_eval=False, adaptive_mode=False,
        #     cutoff=1e-10, merge_threshold=1000)
        mps_module = MPS(
            input_size, output_dim, bond_dim, periodic_bc=bc, label_site=label_site
        )
        assert len(list(mps_module.parameters())) == num_params

        output = mps_module(input_data)
        assert list(output.shape) == [batch_size, output_dim]
