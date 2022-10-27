import logging
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from anode.discrete_models import ResNet
from anode.models import ODENet
from experiments.dataloaders import Data1D, ConcentricSphere
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from phd_experiments.tde.tde_model import TensorODEBLOCK

MODEL_NAMES = ['resnet', 'node', 'anode', 'tode']
DATASETS_NAMES = ['flip1d', 'concentric-sphere']


def get_model(configs: dict):
    if configs['model-name'] not in MODEL_NAMES:
        raise ValueError(f"""Model name {configs['model-name']} is not supported : must be one of {MODEL_NAMES}""")
    if configs['model-name'] == 'resnet':
        return ResNet(data_dim=configs[configs['dataset-name']]['input_dim'],
                      hidden_dim=configs[configs['model-name']]['hidden_dim'],
                      num_layers=configs[configs['model-name']]['num_layers'],
                      output_dim=configs[configs['dataset-name']]['output_dim'])
    elif configs['model-name'] == 'node':
        # augment_dim = 0
        return ODENet(device=torch.device(configs['torch']['device']),
                      data_dim=configs[configs['dataset-name']]['input_dim'],
                      hidden_dim=configs[configs['model-name']]['hidden_dim'],
                      output_dim=configs[configs['dataset-name']]['output_dim'])

    elif configs['model-name'] == 'anode':
        return ODENet(device=torch.device(configs['torch']['device']),
                      data_dim=configs[configs['dataset-name']]['input_dim'],
                      hidden_dim=configs[configs['model-name']]['hidden_dim'],
                      output_dim=configs[configs['dataset-name']]['output_dim'],
                      augment_dim=configs[configs['model-name']]['augment_dim'])
    elif configs['model-name'] == 'tode':
        input_dim = configs[configs['dataset-name']]['input_dim']
        tensor_dims = configs[configs['model-name']]['tensor_dims'][input_dim]
        return TensorODEBLOCK(input_dimensions=[input_dim],
                              output_dimensions=[configs[configs['dataset-name']]['output_dim']],
                              tensor_dimensions=tensor_dims,
                              t_span=(0, 1))
        # TODO should t_span be parameterized ?


def get_data_loader(dataset_name: str, configs: dict):
    if dataset_name == 'flip1d':
        train_dataset = Data1D(num_points=configs[dataset_name]['n_train'], target_flip=configs[dataset_name]['flip'])
        test_dataset = Data1D(num_points=configs[dataset_name]['n_test'], target_flip=configs[dataset_name]['flip'])
    elif dataset_name == 'concentric-sphere':
        inner_range_tuple = tuple(map(float, configs[configs['dataset-name']]['inner_range'].split(',')))
        outer_range_tuple = tuple(map(float, configs[configs['dataset-name']]['outer_range'].split(',')))
        train_dataset = ConcentricSphere(dim=configs[configs['dataset-name']]['input_dim'],
                                         inner_range=inner_range_tuple,
                                         outer_range=outer_range_tuple,
                                         num_points_inner=configs[configs['dataset-name']]['n_inner_train'],
                                         num_points_outer=configs[configs['dataset-name']]['n_outer_train'])
        test_dataset = ConcentricSphere(dim=configs[configs['dataset-name']]['input_dim'],
                                        inner_range=inner_range_tuple,
                                        outer_range=outer_range_tuple,
                                        num_points_inner=configs[configs['dataset-name']]['n_inner_train'],
                                        num_points_outer=configs[configs['dataset-name']]['n_outer_train'])
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
        configs_ = yaml.load(stream=f, Loader=yaml.FullLoader)
    logger.info(f"""Experimenting with model {configs_['model-name']} and dataset {configs_['dataset-name']}""")
    model_ = get_model(configs=configs_)
    train_dataloader_, test_dataloader_ = get_data_loader(dataset_name=configs_['dataset-name'], configs=configs_)
    loss_fn = get_loss(loss_name=configs_['train']['loss'])
    target_flip = True
    optimizer = Adam(model_.parameters(), lr=1e-3)
    loss = torch.tensor([np.Inf])
    epochs_loss_history = []
    logger.info(f"""Starting training with n_epochs = {configs_['train']['n_epochs']},loss_threshold {
        configs_['train']['loss_threshold']} and init loss = {loss.item()}""")
    epoch = None
    for epoch in tqdm(range(1, configs_['train']['n_epochs'] + 1), desc="Epochs"):
        batch_losses = []
        for batch_idx, (X, Y) in enumerate(train_dataloader_):
            optimizer.zero_grad()
            Y_pred = model_(X)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        # print every freq epochs
        if epoch % 10 == 0:
            logger.info(f'epoch = {epoch} | loss = {epoch_loss}')
        epochs_loss_history.append(epoch_loss)
        effective_window = min(len(epochs_loss_history), configs_['train']['loss_window'])
        rolling_avg_loss = np.mean(epochs_loss_history[-effective_window:])
        if rolling_avg_loss <= float(configs_['train']['loss_threshold']):
            logger.info(
                f"""Training ended successfully :
                Rolling average loss = {rolling_avg_loss} <= 
                loss_threshold = {configs_['train']['loss_threshold']}""")
            break
    logger.info(f'final epoch loss = {epochs_loss_history[-1]} at epoch = {epoch}')

    # to make sure pytorch computation graph is freed
    # 
    del loss
    del epochs_loss_history
    del batch_losses
    del model_
    del train_dataloader_
