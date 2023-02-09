import torch
from torch.nn import ParameterList, Parameter, Linear

from dlra.tt import TensorTrain
from phd_experiments.tt_ode.ttode_model import TensorTrainODEBLOCK


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tt_rank = 3
        D_z = 4
        deg = 2
        # self.L = ParameterList(values=[torch.Tensor([2,2])]*2)
        # self.P = Parameter(data=torch.Tensor([2, 2]))
        # values_ = [TensorTrainODEBLOCK.get_tt(ranks=[tt_rank] * int(D_z),
        #                                                           basis_dim=deg, requires_grad=True)] * int(D_z)
        # self.W = ParameterList(values=values_)
        cores = TensorTrainODEBLOCK.generate_tt_cores([tt_rank] * (D_z - 1), basis_dim=deg, requires_grad=True)

        self.W = [TensorTrain(dims=[deg]*D_z,comp_list=cores)]*2
        # self.L = ParameterList(values=[Linear(in_features=3, out_features=4)]*5)
        # n = len(self.L)
        # print(f'len W = {n}')
    def get_W(self):
        return self.W


if __name__ == '__main__':

    model = DummyModel()
    params = model.parameters()
    counter = 0
    W = model.get_W()
    params_to_opt = []
    for w in W:
        p_ = w.parameters()
        for p2 in p_:
            params_to_opt.append(p2)
    print(len(params_to_opt))


    # code snippet to test requires grad and is leaf
    u = torch.distributions.Uniform(low=0.1, high=0.9)
    X = u.sample(sample_shape=torch.Size([2]))
    A = torch.nn.Parameter(u.sample(sample_shape=torch.Size([3, 2])), requires_grad=True)
    B = torch.nn.Parameter(u.sample(sample_shape=torch.Size([3])), requires_grad=True)
    Y = torch.einsum('ij,j->i', A, X) + B
    ###
    print('---')
    print(f'X is leaf = {X.is_leaf}')
    print(f'A is leaf = {A.is_leaf}')
    print(f'B is leaf = {B.is_leaf}')
    print(f'Y is leaf = {Y.is_leaf}')
    ###
    print('---')
    print(f'X requires grad = {X.requires_grad}')
    print(f'A required grad = {A.requires_grad}')
    print(f'B requires grad = {B.requires_grad}')
    print(f'Y requires grad = {Y.requires_grad}')

    A_true = torch.tensor(data=[[1, 2], [3, 4]])
    B_true = torch.tensor(data=[20, 30])
    A_trainable = torch.nn.Parameter(u.sample(sample_shape=torch.Size(A_true.size())))
    B_trainable = torch.nn.Parameter(u.sample(sample_shape=torch.Size(B_true.size())))
    batch_size = 64
    n_batches = 100
    n_epochs = 10
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in range(n_epochs):
        for b in range(n_batches):
            t = A_trainable.size()
            X_true_sample = u.sample(sample_shape=torch.Size([batch_size, list(A_trainable.size())[1]]))
            Y_true_sample = torch.einsum('bj,ij->bi', X_true_sample, A_trainable) + B_trainable

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(Y_true_sample)

    print('finished')
