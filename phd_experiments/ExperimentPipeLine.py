import logging

import torch

from anode.models import ODENet
from anode.training import Trainer
from experiments.dataloaders import ConcentricSphere
from torch.utils.data import DataLoader

from phd_experiments.tde.tde_model import TensorDiffEq
from phd_experiments.tde.tde_training import TensorDiffEqTrainer
from viz.plots import single_feature_plt, get_feature_history, multi_feature_plt, input_space_plt, iteration_plt
from viz.plots import trajectory_plt


class ExperimentPipeline:
    def __init__(self, data_dim, device):

        self.test_data_loader = None
        self.tensor_diff_eq_trainer = None
        self.neural_ode_trainer = None
        self.train_data_loader = None
        self.data_dim = data_dim
        self.device = device
        self.logger = logging.getLogger()

    def load_data(self, num_points_inner=2000, num_points_outer=4000, batch_size=128):
        data_concentric = ConcentricSphere(self.data_dim, inner_range=(0., .5), outer_range=(1., 1.5),
                                           num_points_inner=num_points_inner, num_points_outer=num_points_outer)
        N = len(data_concentric)
        train_size = int(0.8 * N)
        test_size = N - train_size
        data_splits = torch.utils.data.dataset.random_split(dataset=data_concentric, lengths=[train_size, test_size])
        self.train_data_loader = DataLoader(data_splits[0], batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(data_splits[1], batch_size=batch_size, shuffle=True)

    def viz_data(self, batch_size=1024, shuffle=True, viz_file='./data_centric.png'):
        dataloader_viz = DataLoader(self.train_data_loader.dataset, batch_size=batch_size, shuffle=shuffle)
        for inputs, targets in dataloader_viz:
            break
        single_feature_plt(inputs, targets, save_fig=viz_file)

    def train_neural_ode(self, hidden_dim=32, non_linearity='relu', lr=1e-3, num_epochs=50, visualize_features=True,
                         visualize_training=True):
        model = ODENet(device, self.data_dim, hidden_dim, time_dependent=True,
                       non_linearity=non_linearity)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Set up trainer
        self.neural_ode_trainer = Trainer(model, optimizer, self.device)
        for inputs, targets in self.train_data_loader:
            break
        if visualize_features:
            feature_history = get_feature_history(self.neural_ode_trainer, self.train_data_loader, inputs,
                                                  targets, num_epochs)
        else:
            # If we don't record feature evolution, simply train model
            self.neural_ode_trainer.train(self.train_data_loader, num_epochs)
        multi_feature_plt(feature_history[::2], targets, save_fig='feature_history.png')
        for small_inputs, small_targets in self.train_data_loader:
            break
        if visualize_training:
            trajectory_plt(model, small_inputs, small_targets, timesteps=10, save_fig='trajectory.png')
            input_space_plt(model, save_fig='input_space_plt.jpg')
            iteration_plt(histories=experiment_sphere.neural_ode_trainer.histories, y='loss',
                          save_fig='neural_odes_loss.png')
            iteration_plt(histories=experiment_sphere.neural_ode_trainer.histories, y='nfe',
                          save_fig='neural_odes_nfes.png')

    def train_tensor_ode(self, lr=1e-3, num_epochs=100):
        tensor_diff_eq_model = TensorDiffEq(input_dimensions=self.data_dim, output_dimensions=1, t_span=(0, 1),
                                            tensor_dimensions=[2, 2, 2])
        optimizer = torch.optim.Adam(params=tensor_diff_eq_model.parameters(), lr=lr)
        loss_func = torch.nn.SmoothL1Loss()
        self.tensor_diff_eq_trainer = TensorDiffEqTrainer(model=tensor_diff_eq_model, optimizer=optimizer,
                                                          train_data_loader=self.train_data_loader,
                                                          test_data_loader=self.test_data_loader, device=self.device,
                                                          loss_func=loss_func, num_epochs=num_epochs, lr=1e-3,
                                                          print_freq=20)

        self.tensor_diff_eq_trainer.train(model=tensor_diff_eq_model, verbose=True)
        avg_loss = self.tensor_diff_eq_trainer.evaluate(model=tensor_diff_eq_model)
        self.logger.info(f'average loss = {avg_loss}')
        # model diagnosis
        self.tensor_diff_eq_trainer.diagnose(model=tensor_diff_eq_model)


if __name__ == '__main__':
    # Config
    logging.basicConfig(level=logging.DEBUG)
    data_dim = 2
    device = torch.device('cpu')
    #########
    experiment_sphere = ExperimentPipeline(data_dim=data_dim, device=device)
    experiment_sphere.load_data()
    experiment_sphere.viz_data()
    # experiment_sphere.train_neural_ode()
    experiment_sphere.train_tensor_ode()
    ########
