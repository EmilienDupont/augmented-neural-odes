#!/usr/bin/env python3
import torch
import sys

sys.path.append("..")
from contractables import *
from torchmps import *

batch_size = 16
size = 10
D = 3
d = 2

# Check that MatRegion.reduce() works and outputs the correct shape
mats = torch.randn([batch_size, size, D, D])
mat_region = MatRegion(mats)
assert list(mat_region.reduce().tensor.shape) == [batch_size, D, D]

# Check that PeriodicBC.reduce() works and gives basically the same
# answer for left-to-right and right-to-left evaluation
# periodic_bc = PeriodicBC([mat_region]).reduce()
# periodic_bc_rtl = PeriodicBC([mat_region]).reduce(right_to_left=True)
# assert list(periodic_bc.shape) == [batch_size]
# max_diff = float(torch.max(torch.abs(periodic_bc - periodic_bc_rtl)))
# if max_diff > 0:
#    print("Max diff between left and right contractions (PeriodicBC):",
#          max_diff)

# Check that OpenBC.reduce() works and gives basically the same
# answer for left-to-right and right-to-left evaluation
# open_bc = OpenBC([mat_region]).reduce()
# open_bc_rtl = OpenBC([mat_region]).reduce(right_to_left=True)
# assert list(open_bc.shape) == [batch_size]
# max_diff = float(torch.max(torch.abs(open_bc - open_bc_rtl)))
# if max_diff > 0:
#    print("Max diff between left and right contractions (OpenBC):",
#          max_diff)
