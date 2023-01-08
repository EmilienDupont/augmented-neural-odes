import string
from typing import List

import torch

from phd_experiments.tn.tt import TensorTrainFixedRank


def gen_einsum_eqn(n_basis):
    assert n_basis <= len(string.ascii_letters)
    # ab are reserved for output dim and batch indices
    # assume output is a vector
    eqn = "a"
    eqn += string.ascii_letters[2:(n_basis + 2)] + ","
    n = len(eqn) - 2
    eqn += ','.join([f"b{eqn[i]}" for i in range(1, n + 1)]) + "->ba"
    return eqn


def full_weight_tensor_contract(W: torch.Tensor, Phi: List[torch.Tensor]):
    assert len(W.size()) - 1 == len(Phi)
    eqn = gen_einsum_eqn(len(Phi))
    operands = [W] + Phi
    v = torch.einsum(eqn, operands)
    return v


def tt_weight_contract(W: TensorTrainFixedRank, Phi: List[torch.Tensor]):
    pass


def poly2(x: torch.Tensor, t: float, degree: int):
    """
    x : torch.Tensor of size Batch x dim
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.size()) == 2, "Supporting poly-basis generation for only torch-vectors"
    B = x.size()[0]
    x = torch.cat([x, torch.tensor(t, dtype=x.dtype).repeat(B).view(-1, 1)], dim=1)
    pow_tensors = [torch.ones(x.size()), x]
    for deg in range(2, degree + 1):
        pow_tensors.append(torch.pow(x, deg))
    Phi = list(torch.permute(input=torch.stack(tensors=pow_tensors, dim=0), dims=[2, 1, 0]))
    return Phi


def expand_contract(C, x):
    n = 3
    rows_ = []
    rows_.append(torch.ones(list(x.size())[0]))
    rows_.append(x)
    for i in range(2, n + 1):
        rows_.append(torch.pow(x, i))

    tns = torch.stack(rows_).T
    list_ = list(tns)
    list_.insert(0, C)
    eqn = "ijklq,j,k,l,q->i"
    r = torch.einsum(eqn, list_)
    return r


if __name__ == '__main__':
    # t_ = torch.tensor([[1, 2], [3, 4]])
    # l_ = list(t_)
    # a = torch.tensor([1., 2., 3.])
    # b = torch.tensor([5., 6., 7.])
    # c = torch.tensor([9., 10., 11.])
    # eqn = "i,j,k->ijk"
    # operands = [a, b, c]
    # r = torch.einsum(eqn, operands)
    #
    # print(r.size())
    # print(r)
    #
    # ####
    #
    # C = torch.ones([4, 5, 5], requires_grad=True)
    # # C_B = C.unsqueeze(0).repeat([64, 1, 1, 1, 1, 1])
    # # print(C_B.requires_grad)
    # operands = [C, torch.ones(64, 5), torch.ones(64, 5)]
    # # eqn = "aijkl,bi,bj,bk,bl->ba"
    # eqn = "aij,bi,bj->ba"
    # r = torch.einsum(eqn, operands)
    # print(r)
    # e_ = torch.pow(input=torch.tensor([1, 2]), exponent=torch.tensor([0, 1, 2, 3]))
    # print(e_)
    #
    batch = 64
    dim = 6
    deg = 3
    x = torch.rand(batch, dim)
    Phi = poly2(x=x, t=1, degree=deg)
    C_size = [dim] + [deg + 1 for _ in range(dim + 1)]
    C = torch.rand(C_size)
    full_weight_tensor_contract(W=C, Phi=Phi)
    print("")
