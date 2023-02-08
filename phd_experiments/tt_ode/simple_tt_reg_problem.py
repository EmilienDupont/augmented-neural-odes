"""
The purpose of this script is to experiment
1- ALS training
2- Gradient descent training ( as a base-line)
for a simple sklearn regression problem
Why? I want to set a several baselines
1- TT-ODE trained by normal Backprop-Gradient-Descent
2- TT-ODE trained by Adjoint-Sensitivity

refs
https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch import Tensor
from torch.nn import Sequential, Linear, MSELoss
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dlra.tt import TensorTrain
from phd_experiments.tt_ode.basis import Basis
from phd_experiments.tt_ode.ttode_model import TensorTrainODEBLOCK


class SklearnDiabetesDataset(Dataset):

    def __init__(self, dtype: torch.dtype):
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        self.x_train = torch.tensor(torch.Tensor(X.values), dtype=dtype)
        self.y_train = torch.tensor(torch.Tensor(y.values), dtype=dtype)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class NN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_dim, hidden_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_dim, hidden_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor):
        y = self.model(x)
        return y


import pandas
from sklearn.datasets import load_diabetes

if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger()
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    logger.info(f'X shape = {X.shape}')
    logger.info(f'Y shape = {y.shape}')
    # training params

    epochs = 1000
    batch_size = 64
    hidden_dim = 256
    model_type = 'tt'
    basis_poly_deg = 4
    dtype = torch.float64
    # training
    ds = SklearnDiabetesDataset(dtype=dtype)
    train_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    model = None
    if model_type == 'nn':
        model = NN(input_dim=X.shape[1], out_dim=1, hidden_dim=hidden_dim)
    else:
        model = TensorTrainODEBLOCK.get_tt(ranks=[3] * (X.shape[1] - 1), basis_dim=basis_poly_deg + 1,
                                           requires_grad=True, dtype=dtype)
    parameters_ = model.parameters()
    parameters_list = list(parameters_)
    if model_type == 'tt':
        assert len(parameters_list) == X.shape[1], "Number of cores (tt parameters) must = data_dim"
    optimizer = Adam(params=parameters_list, lr=1e-3)
    mse_fn = MSELoss()
    epochs_avg_losses = []
    for epoch in tqdm(range(epochs),desc='epoch'):
        epoch_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = None
            if model_type in ['nn']:
                y_hat = model(X)
            elif model_type == 'tt':
                Phi = Basis.poly(x=X, t=None, poly_deg=basis_poly_deg)
                y_hat = model(Phi)
            assert y_hat is not None, "y_hat cannot be None"
            mse_val = mse_fn(y, y_hat)
            rmse_val = torch.sqrt(mse_val)
            epoch_losses.append(rmse_val.item())
            mse_val.backward()

            if isinstance(model, TensorTrain):
                tt_norm_before = model.norm()
            optimizer.step()
            if isinstance(model, TensorTrain):
                tt_norm_after = model.norm()
        epochs_avg_losses.append(np.nanmean(epoch_losses))
        if epoch % 10 == 0:
            logger.info(f'Epoch : {epoch} => rmse = {epoch_losses[-1]}')
            if isinstance(model, TensorTrain):
                logger.info(f'TT-Norm delta = {tt_norm_after - tt_norm_before}')

    assert len(epochs_avg_losses) == epochs
    print(len(epochs_avg_losses))
    # TODO compare results with
    #   https://www.kaggle.com/code/rahulrajpandey31/diabetes-analysis-linear-reg-from-scratch/notebook
    logger.info('training finished')
    fig = plt.plot(epochs_avg_losses)
    plt.savefig(f'convergence_{model_type}.png')
