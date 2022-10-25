import logging
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from anode.discrete_models import ResNet
from anode.models import ODENet
from experiments.dataloaders import Data1D
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from phd_experiments.tde.tde_model import TensorODEBLOCK

MODEL_NAMES = ['resnet', 'neuralode', 'anode', 'tensorode']
DATASETS_NAMES = ['flip1d']


def get_model(model_name: str, configs: dict):
    if model_name == 'resnet':
        return ResNet(data_dim=configs['data']['data_dim'], hidden_dim=configs[model_name]['hidden_dim'],
                      num_layers=configs[model_name]['num_layers'],
                      output_dim=configs['data']['out_dim'])
    elif model_name == 'neuralode':
        # augment_dim = 0
        return ODENet(device=torch.device(configs['torch']['device']), data_dim=configs['data']['data_dim'],
                      hidden_dim=configs[model_name]['hidden_dim'], output_dim=configs['data']['out_dim'])

    elif model_name == 'anode':
        return ODENet(device=torch.device(configs['torch']['device']), data_dim=configs['data']['data_dim'],
                      hidden_dim=configs[model_name]['hidden_dim'], output_dim=configs['data']['out_dim'],
                      augment_dim=configs[model_name]['augment_dim'])
    elif model_name == 'tensorode':
        return TensorODEBLOCK(input_dimensions=[configs['data']['data_dim']],
                              output_dimensions=[configs['data']['out_dim']],
                              tensor_dimensions=configs[model_name]['tensor_dims'],
                              t_span=(0, 1))
        # TODO should t_span be parameterized ?
    else:
        raise ValueError(f'Model name {model_name} is not supported : must be one of {MODEL_NAMES}')


def get_data_loader(dataset_name: str, configs: dict):
    if dataset_name == 'flip1d':
        train_dataset = Data1D(num_points=configs['data']['n_train'], target_flip=configs[dataset_name]['flip'])
        test_dataset = Data1D(num_points=configs['data']['n_test'], target_flip=configs[dataset_name]['flip'])
    else:
        raise ValueError(f'data {dataset_name} is not supported ! ')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=configs['train']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=configs['train']['batch_size'], shuffle=True)
    return train_dataloader, test_dataloader


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config file')
    return parser


def get_loss(loss_name: str):
    if loss_name == 'smoothl1loss':
        return SmoothL1Loss()


if __name__ == '__main__':
    # TODO
    """
    1) why sometime ODENet loss reaches 0 ??
    2) plot loss, loss convergence , nfes for each model
    3) plot learning curve, running time
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    parser = get_parser()
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(stream=f, Loader=yaml.FullLoader)
    logger.info(f"""Experimenting with model : {configs['model']['name']}""")
    model_ = get_model(model_name=configs['model']['name'], configs=configs)
    train_dataloader_, test_dataloader_ = get_data_loader(dataset_name=configs['data']['name'], configs=configs)
    loss_fn = get_loss(loss_name=configs['train']['loss'])
    target_flip = True
    batch_size = 32
    n_epochs = 100
    optimizer = Adam(model_.parameters(), lr=1e-3)
    loss = torch.tensor([np.Inf])
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        for batch_idx, (X, Y) in enumerate(train_dataloader_):
            optimizer.zero_grad()
            Y_pred = model_(X)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
        # print every freq epochs
        if epoch % 10 == 0:
            logger.info(f'epoch = {epoch} | loss = {loss.item()}')
    logger.info(f'final loss = {loss.item()}')
