#!/usr/bin/env python3
import torch
import sys
from random import randint

sys.path.append("..")
from torchmps import TI_MPS

feature_dim = 2
batch_size = 7
seq_length = 4
output_dim = 3
bond_dim = 5

# Generate a random pre-embedded batch input tensor
batch_input = torch.randn([batch_size, seq_length, feature_dim])

# Generate a list of random input sequences with different lengths
seq_input = [torch.randn([randint(1, seq_length)]) for _ in range(batch_size)]
seq_input_emb = [
    torch.randn([randint(1, seq_length), feature_dim]) for _ in range(batch_size)
]

# Base configuration for our test
base_config = {
    "parallel_eval": False,
    "fixed_ends": False,
    "use_bias": True,
    "fixed_bias": True,
}

# Test out multiple configurations
all_configs = [dict(base_config.items()) for _ in range(len(base_config) + 1)]
for i, param in enumerate(base_config.keys(), 1):
    all_configs[i][param] = not base_config[param]

# Try each of the parameter configs in turn and run on both inputs
for config in all_configs:
    mps_module = TI_MPS(output_dim, bond_dim, feature_dim, **config)

    # Feed batch input to our MPS, and check the output size
    batch_output = mps_module(batch_input)
    assert list(batch_output.shape) == [batch_size, output_dim]

    # Sum the outputs and generate gradients
    batch_sum = torch.sum(batch_output)
    batch_sum.backward()

    # Get the parameter tensors and check that all of them have gradients
    for tensor in list(mps_module.parameters()):
        assert tensor.grad is not None

    # Try feeding in our embedded and unembedded variable length inputs, but
    # skip this when our zero-padding rules would throw an error
    if not (config["use_bias"] and config["fixed_bias"]):
        continue

    # Get outputs, check size
    seq_output = mps_module(seq_input)
    seq_output_emb = mps_module(seq_input_emb)
    assert list(seq_output.shape) == [batch_size, output_dim]
    assert list(seq_output_emb.shape) == [batch_size, output_dim]
