import logging

import numpy as np
import torch

from anode.discrete_models import ResNet
from anode.models import ODENet
from experiments.dataloaders import Data1D
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from phd_experiments.tde.tde_model import TensorODEBLOCK

if __name__ == '__main__':
    # TODO
    """
    1) why sometime ODENet loss reaches 0 ??
    2) plot loss, loss convergence , nfes for each model
    3) plot learning curve, running time
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    loss_fn = SmoothL1Loss()
    lr = 1e-3
    epoch_print_freq = 10
    device = torch.device('cpu')
    data_dim = output_dim = 1
    hidden_dim = 32
    #
    target_flip = True
    batch_size = 32
    n_epochs = 100
    dataset_1d_train = Data1D(num_points=1000, target_flip=target_flip)
    dataset_1d_test = Data1D(num_points=200, target_flip=target_flip)
    train_dataloader = DataLoader(dataset=dataset_1d_train, batch_size=batch_size)

    models = [('ResNet', ResNet(data_dim=data_dim, hidden_dim=hidden_dim, num_layers=20), False),
              ('ODENet', ODENet(device=device, data_dim=1, hidden_dim=hidden_dim, output_dim=output_dim,
                                augment_dim=0), False),
              ('ANODE', ODENet(device=device, data_dim=1, hidden_dim=hidden_dim, output_dim=output_dim,
                               augment_dim=1), False),
              ('TensorODE',
               TensorODEBLOCK(input_dimensions=[data_dim], output_dimensions=[output_dim], tensor_dimensions=[2],
                              t_span=(0, 1)), True)]
    # Train ResNet
    for model_name, model, to_train in models:
        if not to_train:
            continue
        logger.info(f'Training Model : {model_name}')
        optimizer = Adam(model.parameters(), lr=lr)
        loss = torch.tensor([np.Inf])
        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            for batch_idx, (X, Y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                Y_pred = model(X)
                loss = loss_fn(Y_pred, Y)
                loss.backward()
                optimizer.step()
            # print every freq epochs
            if epoch % epoch_print_freq == 0:
                logger.info(f'{model_name} | epoch = {epoch} | loss = {loss.item()}')
        logger.info(f'{model_name} final loss = {loss.item()}')

