import argparse
import datetime
import logging
import os.path

import torch
import yaml

from anode.models import ODEFunc, ODEBlock
from anode.training import Trainer
from experiments.dataloaders import Data1D
from torch.utils.data import DataLoader

from viz.plots import vector_field_plt


class Experiment:
    def __init__(self, device, model_type: str, data_loader, num_epochs: int, timestamp: str, results_subdir: str):
        self.ode_func = None
        self.logger = logging.getLogger()
        self.config = config
        self.device = device
        self.model_type = model_type
        self.trainer = None
        self.model = None
        self.optimizer = None
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.timestamp = timestamp
        self.results_subdir = results_subdir

    def init_neural_ode_model(self, lr, data_dim, hidden_dim):
        self.ode_func = ODEFunc(device=device, data_dim=data_dim, hidden_dim=hidden_dim, time_dependent=True)
        self.model = ODEBlock(device=device, odefunc=self.ode_func)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

    def init_model(self, lr, print_freq, data_dim, hidden_dim):
        self.logger.info(f'Initializing model of type : {self.model_type}')
        if self.model_type == 'NeurODE':
            self.init_neural_ode_model(lr=lr, data_dim=data_dim, hidden_dim=hidden_dim)
        else:
            self.logger.error(f'Model type : {self.model_type} not supported yet!')
        self.trainer = Trainer(self.model, self.optimizer, device, print_freq=print_freq)

    def train_model(self):
        self.trainer.train(data_loader=self.data_loader, num_epochs=self.num_epochs)

    def predict(self):
        for inputs, targets in dataloader:
            break
        y = self.model(inputs)
        print(f"y_avg : {torch.mean(y)}")
        print(f"y_std : {torch.std(y)}")

    def viz(self, num_points, timesteps, h_min, h_max):
        for inputs, targets in self.data_loader:
            break
        results_dir = f'{self.results_subdir}_{self.timestamp}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_fig_path = os.path.join(results_dir, f'{self.model_type}.png')
        vector_field_plt(odefunc=self.ode_func, num_points=num_points, timesteps=timesteps,
                         inputs=inputs[:8], targets=targets[:8],
                         h_min=h_min, h_max=h_max, model=self.model,
                         save_fig=save_fig_path)


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
    # TODO
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
    # FIXME for debug, remove
    for inputs, targets in dataloader:
        break
    # TODO =>  1 : Running the ResNet experiment

    # 2: Running the NeurODE experiment
    model_type = 'NeurODE'
    experiment_ = Experiment(device=device, model_type='NeurODE', data_loader=dataloader,
                             num_epochs=config['training']['num_epochs'],
                             results_subdir=config['viz']['subdir'], timestamp=experiment_timestamp)
    experiment_.init_model(lr=config['training']['lr'], print_freq=config['training']['print_freq'],
                           data_dim=config['data']['dim'], hidden_dim=config['training']['hidden_dim'])
    experiment_.train_model()
    experiment_.viz(num_points=config['viz']['num_points'], timesteps=config['viz']['timesteps'],
                    h_min=config['viz']['h_min'], h_max=config['viz']['h_max'])
    experiment_.predict()

