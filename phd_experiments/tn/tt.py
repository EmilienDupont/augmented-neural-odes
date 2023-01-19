import logging
from typing import List

import torch
from torch import Tensor


class TensorTrainFixedRank:
    def __init__(self, order: int, core_input_dim: int, out_dim: int, fixed_rank: int, requires_grad: bool):
        tensor_uniform = torch.distributions.Uniform(low=0.01, high=0.5)
        self.core_tensors = [torch.nn.Parameter(tensor_uniform.sample(torch.Size([core_input_dim, fixed_rank])),
                                                requires_grad=requires_grad)]
        self.out_dim = out_dim
        for i in range(1, order - 1):
            self.core_tensors.append(
                torch.nn.Parameter(
                    tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, core_input_dim, fixed_rank])),
                    requires_grad=requires_grad))
        self.core_tensors.append(
            torch.nn.Parameter(tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, core_input_dim, out_dim])),
                               requires_grad=requires_grad))
        assert len(self.core_tensors) == order, \
            f"# core tensors should == order : {len(self.core_tensors)} != {order}"
        self.logger = logging.getLogger()

    def norm(self):
        tot_norm = 0
        for tensor in self.core_tensors:
            tot_norm += torch.norm(tensor)
        return tot_norm

    def display_sizes(self):
        sizes = []
        for tensor in self.core_tensors:
            sizes.append(tensor.size())
        self.logger.info(f'TensorTrain sizes = {sizes}')

    def is_parameter(self):
        requires_grads = list(map(lambda x: x.requires_grad, self.core_tensors))
        return all(requires_grads)

    def contract_basis(self, basis_tensors: List[Tensor]) -> Tensor:
        assert len(self.core_tensors) == len(
            basis_tensors), f"# of core-tensors must == number of basis tensors " \
                            f", {len(self.core_tensors)} != {len(basis_tensors)}"
        n_cores = len(self.core_tensors)
        # first core
        res_tensor = torch.einsum("ij,bi->bj", self.core_tensors[0], basis_tensors[0])
        # middle cores
        for i in range(1, len(self.core_tensors) - 1):
            core_basis = torch.einsum("ijk,bj->bik", self.core_tensors[i], basis_tensors[i])
            res_tensor = torch.einsum("bi,bik->bk", res_tensor, core_basis)
        # last core
        core_basis = torch.einsum("ijl,bj->bil", self.core_tensors[n_cores - 1], basis_tensors[n_cores - 1])
        res_tensor = torch.einsum("bi,bil->bl", res_tensor, core_basis)
        assert res_tensor.size()[1] == self.out_dim, f"output tensor size must = " \
                                                     f"out_dim : {res_tensor.size()}!={self.out_dim}"
        return res_tensor

    def display_cores(self):
        self.logger.info(f'Cores : \n{self.core_tensors}\n')


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    main_logger = logging.getLogger()
    order = 4
    basis_dim = 3
    out_dim = 10
    fixed_rank = 2

    ttxr = TensorTrainFixedRank(order=order, core_input_dim=basis_dim, out_dim=out_dim, fixed_rank=fixed_rank,
                                requires_grad=True)
    norm_val = ttxr.norm()
    ttxr.display_sizes()
    ttxr.is_parameter()
    main_logger.info(ttxr.is_parameter())

    basis_ = []
    for i in range(order):
        basis_.append(torch.ones(basis_dim))
    res = ttxr.contract_basis(basis_tensors=basis_)
