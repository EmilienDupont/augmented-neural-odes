import datetime
import logging
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import yaml
from torch import Tensor

from anode.discrete_models import ResNet
from anode.models import ODENet
from dlra.tt import TensorTrain
from experiments.dataloaders import Data1D, ConcentricSphere
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from phd_experiments.tn.tt import TensorTrainFixedRank
from phd_experiments.tt_ode.ttode_model import TensorTrainODEBLOCK

MODEL_NAMES = ['resnet', 'node', 'anode', 'ttode']
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
    elif configs['model-name'] == 'ttode':
        input_dim = configs[configs['dataset-name']]['input_dim']
        tensor_dims = configs[configs['model-name']]['tensor_dims'][input_dim]
        non_linearity = None if configs[configs['model-name']]['non_linearity'] == 'None' else \
            configs[configs['model-name']]['non_linearity']
        tt_rank = configs[configs['model-name']]['tt_rank']
        return TensorTrainODEBLOCK(input_dimensions=[input_dim],
                                   output_dimensions=[configs[configs['dataset-name']]['output_dim']],
                                   tensor_dimensions=tensor_dims, basis_str=configs[configs['model-name']]['basis'],
                                   t_span=tuple(configs[configs['model-name']]['t_span']), non_linearity=non_linearity,
                                   forward_impl_method=configs[configs['model-name']]['forward_impl_method'],
                                   tt_rank=tt_rank,
                                   custom_autograd_fn=configs[configs['model-name']]['custom_autograd_fn'])


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


def get_W_norm(W: [Tensor, TensorTrain, TensorTrainFixedRank]) -> Tensor:
    if isinstance(W, (Tensor, TensorTrainFixedRank)):
        W_Norm = W.norm()
    elif isinstance(W, list) and all([isinstance(w, TensorTrain) for w in W]):
        W_Norm = sum([w.norm() for w in W])
    else:
        raise ValueError(f'W is not of supported type , its type is : {type(W)}, and supported types are '
                         f'{[Tensor, TensorTrainFixedRank, List[TensorTrain]]}')
    assert isinstance(W_Norm, Tensor) and W_Norm.size() == torch.Size([]), f"W_norm must be float but of type {W_Norm}"
    return W_Norm


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
    if configs_['model-name'] == 'ttode':
        logger.info(f"""Forward-Integration method is : {configs_[configs_['model-name']]['forward_impl_method']}""")
    model_ = get_model(configs=configs_)
    train_dataloader_, test_dataloader_ = get_data_loader(dataset_name=configs_['dataset-name'], configs=configs_)
    loss_fn = get_loss(loss_name=configs_['train']['loss'])
    target_flip = True

    optimizer = Adam(model_.parameters(), lr=float(configs_['train']['lr']))
    loss = torch.tensor([np.Inf])
    epochs_loss_history = []
    logger.info(f"""Starting training with n_epochs = {configs_['train']['n_epochs']},loss_threshold {
    configs_['train']['loss_threshold']} and init loss = {loss.item()}""")
    epoch = None
    start_time = datetime.datetime.now()
    batch_size = configs_['train']['batch_size']
    forward_call_count = 0
    lambda_ = configs_[configs_['model-name']]['lambda'] if configs_['model-name'] == 'ttode' else 0
    for epoch in tqdm(range(1, configs_['train']['n_epochs'] + 1), desc="Epochs"):
        batch_losses = []
        for batch_idx, (X, Y) in enumerate(train_dataloader_):
            optimizer.zero_grad()
            Y_pred = model_(X)
            forward_call_count += 1
            W_norm = get_W_norm(model_.get_W())
            reg_term = lambda_ * W_norm
            loss = loss_fn(Y_pred, Y) + reg_term
            # call backward hooks
            loss.backward()
            # save old norms
            P_old_norm = model_.get_P().norm().item()
            W_old_norm = get_W_norm(model_.get_W())
            Q_old_norm = model_.get_Q().norm().item()

            optimizer.step()

            # calculate delta norm
            delta_P_norm = model_.get_P().norm().item() - P_old_norm
            # P_diff_opt_manual = torch.norm(P_new_manual - model_.get_P())
            W_new_norm = get_W_norm(model_.get_W())
            delta_W_norm = W_new_norm - W_old_norm
            delta_Q = model_.get_Q().norm().item() - Q_old_norm
            # Log model state after optimize.step()

            # if configs_['model-name'] == 'ttode' and configs_['ttode']['forward_impl_method'] == 'ttode_als':
            #     # apply the tt-als logic for P and W . optimizer.step() should have taken care of terminal_nn
            #     if configs_['model-name'] == 'ttode' and
            #     configs_['ttode']['custom_autograd_fn'] == True:
            #         with torch.no_grad():
            #             pass
            #     else:
            #         pass

            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        # print every freq epochs
        if epoch % 10 == 0:
            logger.info(f'epoch = {epoch} | loss = {epoch_loss} | P_norm_delta = {delta_P_norm} , W_norm_delta = '
                        f'{delta_W_norm} , Q_delta = {delta_Q}')

        epochs_loss_history.append(epoch_loss)
        effective_window = min(len(epochs_loss_history), configs_['train']['loss_window'])
        rolling_avg_loss = np.mean(epochs_loss_history[-effective_window:])
        if rolling_avg_loss <= float(configs_['train']['loss_threshold']):
            logger.info(
                f"""Training ended successfully :
                Rolling average loss = {rolling_avg_loss} <= 
                loss_threshold = {configs_['train']['loss_threshold']}""")
            break
    if isinstance(model_, TensorTrainODEBLOCK):
        logger.info(f'for TT-ODE model : \n'
                    f'P = {model_.get_P()}\n'
                    f'F = {model_.get_Q()}\n'
                    f'M = {model_.get_W()}\n')
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    total_nfe = model_.get_nfe() if isinstance(model_, (ODENet, TensorTrainODEBLOCK)) else None
    # avg_nfe = float(model_.get_nfe())/forward_call_count if isinstance(model_, (ODENet, TensorODEBLOCK)) else None
    logger.info(
        f'final epoch loss = {epochs_loss_history[-1]} at epoch = {epoch}\n'
        f'training time = {training_time.seconds} seconds\n'
        f'total_nfe(@num_epochs={epoch}) = {total_nfe}\n')
