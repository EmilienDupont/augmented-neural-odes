#!/usr/bin/env python3
import torch
import sys

sys.path.append("..")
from torchmps import MPS

# Get a reference input to our map
std_input = torch.randn([7, 10])

# Define a couple custom feature maps
def feature_map2(inp):
    return torch.tensor([inp, 1 - inp])


def feature_map3(inp):
    return torch.tensor([inp, 1 - inp, 2 - inp])


my_mps2 = MPS(10, 5, 2, feature_dim=2)
default_output2 = my_mps2(std_input)

my_mps2.register_feature_map(feature_map2)
custom_output2 = my_mps2(std_input)

# Compare the output from the default feature map with that from our custom
# equivalent to that map, which should be identical
assert torch.all(default_output2 == custom_output2)

# Make sure we don't get any errors when using a feature map with higher
# feature dim
my_mps3 = MPS(10, 5, 2, feature_dim=3)
my_mps3.register_feature_map(feature_map3)
output = my_mps3(std_input)

# Make sure pre-embedded inputs are accepted without any errors
output = my_mps2(torch.randn([7, 10, 2]))
output = my_mps3(torch.randn([7, 10, 3]))
