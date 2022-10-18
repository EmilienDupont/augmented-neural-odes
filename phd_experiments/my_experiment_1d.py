import argparse
import datetime
import logging
import os.path
import pandas as pd
import torch
import yaml
import numpy as np
from anode.models import ODEFunc, ODEBlock, ODENet
from anode.training import Trainer
from experiments.dataloaders import Data1D
from torch.utils.data import DataLoader
from anode.discrete_models import ResNet
from viz.plots import vector_field_plt, anode_plt


class Experiment_1D:
    def __init__(self, device, model_name: str, data_loader, num_epochs: int, timestamp: str, results_subdir: str):
        self.ode_func = None
        self.logger = logging.getLogger()
        self.config = config
        self.device = device
        self.model_name = model_name
        self.trainer = None
        self.model = None
        self.optimizer = None
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.timestamp = timestamp
        self.results_subdir = results_subdir

    def init_neural_ode_model(self, data_dim, hidden_dim):
        ode_func = ODEFunc(device=self.device, data_dim=data_dim, hidden_dim=hidden_dim, time_dependent=True)
        self.model = ODEBlock(device=self.device, odefunc=ode_func)

    def init_resnet_model(self, data_dim, hidden_dim, num_layers):
        self.model = ResNet(data_dim=data_dim, hidden_dim=hidden_dim, output_dim=data_dim, num_layers=num_layers)

    def init_anode_model(self, data_dim, hidden_dim, augment_dim, non_linearity):
        self.model = ODENet(device=self.device, data_dim=data_dim, hidden_dim=hidden_dim, output_dim=data_dim,
                            augment_dim=augment_dim,
                            non_linearity=non_linearity)

    def init_model(self, data_dim, hidden_dim, lr, print_freq, **kwargs):
        self.logger.info(f'Initializing model of type : {self.model_name}')
        if self.model_name == 'node':
            self.init_neural_ode_model(data_dim=data_dim, hidden_dim=hidden_dim)
        elif self.model_name == 'resnet':
            self.init_resnet_model(data_dim=data_dim, hidden_dim=hidden_dim, num_layers=kwargs['num_layers'])
        elif self.model_name == 'anode':
            self.init_anode_model(data_dim=data_dim, hidden_dim=hidden_dim, augment_dim=kwargs['augment_dim'],
                                  non_linearity=kwargs['non_linearity'])
        else:
            raise ValueError(f'Model type : {self.model_name} not supported !')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.trainer = Trainer(self.model, self.optimizer, device, print_freq=print_freq)

    def train_model(self):
        self.trainer.train(data_loader=self.data_loader, num_epochs=self.num_epochs)

    def predict(self):
        for inputs, targets in dataloader:
            break
        y_hat = self.model(inputs)
        y_hat_np = np.transpose(y_hat.cpu().detach().numpy()).tolist()[0]
        targets_np = np.transpose(targets.cpu().detach().numpy()).tolist()[0]
        df = pd.DataFrame({'y_np': y_hat_np, 'target_np': targets_np})
        print(f"y_avg : {torch.mean(y_hat)}")
        print(f"y_std : {torch.std(y_hat)}")

    def viz(self, model, odefunc_or_res_blocks, num_points, timesteps, h_min, h_max):
        for inputs, targets in self.data_loader:
            break
        results_dir = f'{self.results_subdir}_{self.timestamp}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_fig_path = os.path.join(results_dir, f'{self.model_name}.png')
        if self.model_name in ['resnet', 'node']:
            vector_field_plt(odefunc=odefunc_or_res_blocks, num_points=num_points, timesteps=timesteps,
                             inputs=inputs[:8], targets=targets[:8],
                             h_min=h_min, h_max=h_max, model=model,
                             save_fig=save_fig_path)
        elif self.model_name == 'anode':
            anode_plt(model=model, num_points=num_points, timesteps=timesteps, h_min=h_min, h_max=h_max,
                      inputs=inputs[:8], targets=targets[:8],
                      save_fig=save_fig_path)

    @staticmethod
    def run_experiment_model(model_name: str, config: dict):
        logger = logging.getLogger()
        logger.info(f'Running experiment for model : {model_name}')
        experiment_ = Experiment_1D(device=device, model_name=model_name, data_loader=dataloader,
                                    num_epochs=config['training']['num_epochs'],
                                    results_subdir=config['viz']['subdir'], timestamp=experiment_timestamp)
        if model_name == 'resnet':
            experiment_.init_model(lr=config['training']['lr'], print_freq=config['training']['print_freq'],
                                   data_dim=config['data']['dim'], hidden_dim=config['training']['hidden_dim'],
                                   num_layers=config['resnet']['num_layers'])
        elif model_name == 'node':
            experiment_.init_model(lr=config['training']['lr'], print_freq=config['training']['print_freq'],
                                   data_dim=config['data']['dim'], hidden_dim=config['training']['hidden_dim'])
        elif model_name == 'anode':
            experiment_.init_model(lr=config['training']['lr'], print_freq=config['training']['print_freq'],
                                   data_dim=config['data']['dim'], hidden_dim=config['training']['hidden_dim'],
                                   augment_dim=config['anode']['augment_dim'],
                                   non_linearity=config['anode']['non_linearity'])
        else:
            raise ValueError(f'model_name {model_name} not supported')
        experiment_.train_model()
        experiment_.predict()
        if model_name in ['node']:
            experiment_.viz(model=experiment_.model, odefunc_or_res_blocks=experiment_.model.odefunc,
                            num_points=config['viz']['num_points'],
                            timesteps=config['resnet']['num_layers'],
                            h_min=config['viz']['h_min'], h_max=config['viz']['h_max'])
        elif model_name == 'resnet':
            experiment_.viz(model=experiment_.model, odefunc_or_res_blocks=experiment_.model.residual_blocks,
                            num_points=config['viz']['num_points'],
                            timesteps=config['resnet']['num_layers'],
                            h_min=config['viz']['h_min'], h_max=config['viz']['h_max'])
        elif model_name == 'anode':
            experiment_.viz(model=experiment_.model, odefunc_or_res_blocks=None,
                            num_points=config['viz']['num_points'],
                            timesteps=config['resnet']['num_layers'],
                            h_min=config['viz']['h_min'], h_max=config['viz']['h_max'])
        else:
            logger.error(f'Plotting for model : {model_name} is not supported !')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser


def config_load(config_file):
    logger = logging.getLogger()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        logger.info(f'Loaded config = {config}')
    return config


if __name__ == '__main__':
    # TODO : In General
    """
    Reproduce 1d results in paper
    Sec 3 
    https://papers.nips.cc/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf
    Fig. 3
    Models: 
    i) RESNET
    ii) NeurODE
    iii) ANODE
    iv) TTDE
    """
    # TODO NOW

    """
    1. Verify resnet predictions. Done
    2. Plot resnet trajectory for the 1d hard problem. Done 
    3. Apply Anode to the 1d hard problem and debug each step in detail
    4. TTDE ?
    """
    # TODO Later
    """
    1. check gradient arrows for resnet dhdt vs dtdt ; dhdt to much zeros and velocity plot is a horizontal line ??
    """
    # args and configs
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config
    config = config_load(config_file)
    device = torch.device(config['device'])
    experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # data
    data = Data1D(num_points=config['data']['num_points'], target_flip=bool(config['data']['flipped']))
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    for inputs, targets in dataloader:
        break
    # model_names = ['node', 'resnet', 'anode', 'ttde']
    model_names = ['anode']
    for model_name in model_names:
        Experiment_1D.run_experiment_model(model_name=model_name, config=config)
