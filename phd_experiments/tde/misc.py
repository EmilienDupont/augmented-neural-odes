import torch

import einops


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
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([5., 6., 7.])
    c = torch.tensor([9., 10., 11.])
    eqn = "i,j,k->ijk"
    operands = [a, b, c]
    r = torch.einsum(eqn, operands)

    print(r.size())
    print(r)

    ####
    C = torch.ones([4, 5, 5], requires_grad=True)
    # C_B = C.unsqueeze(0).repeat([64, 1, 1, 1, 1, 1])
    # print(C_B.requires_grad)
    operands = [C, torch.ones(64, 5), torch.ones(64, 5)]
    # eqn = "aijkl,bi,bj,bk,bl->ba"
    eqn = "aij,bi,bj->ba"
    r = torch.einsum(eqn, operands)
    print(r)
    e_ = torch.pow(input=torch.tensor([1, 2]), exponent=torch.tensor([0, 1, 2, 3]))
    print(e_)
