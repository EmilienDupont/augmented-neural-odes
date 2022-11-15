import string
from typing import List

import torch


def gen_einsum_eqn(n_basis):
    assert n_basis <= len(string.ascii_letters)
    # ab are reserved for output dim and batch indices
    # assume output is a vector
    eqn = "a"
    eqn += string.ascii_letters[2:(n_basis + 2)] + ","
    n = len(eqn) - 2
    eqn += ','.join([f"b{eqn[i]}" for i in range(1, n + 1)]) + "->ba"
    return eqn


def func_tensors_contract(C: torch.Tensor, Phi: List[torch.Tensor]):
    assert len(C.size()) - 1 == len(Phi)
    eqn = gen_einsum_eqn(len(Phi))
    operands = [C] + Phi
    v = torch.einsum(eqn, operands)
    return v
