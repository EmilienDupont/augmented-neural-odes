import torch

from anode.models import ODENet
from anode.training import Trainer
from experiments.dataloaders import ConcentricSphere
from torch.utils.data import DataLoader
from viz.plots import single_feature_plt, get_feature_history, multi_feature_plt


class ExperimentSphere:
    def __init__(self, data_dim, device):
        self.data_loader = None
        self.data_dim = data_dim
        self.device = device

    def load_data(self, num_points_inner=1000, num_points_outer=2000, batch_size=128):
        data_concentric = ConcentricSphere(self.data_dim, inner_range=(0., .5), outer_range=(1., 1.5),
                                           num_points_inner=num_points_inner, num_points_outer=num_points_outer)
        self.data_loader = DataLoader(data_concentric, batch_size=batch_size, shuffle=True)

    def viz_data(self, batch_size=1024, shuffle=True, viz_file='./data_centric.png'):
        dataloader_viz = DataLoader(self.data_loader.dataset, batch_size=batch_size, shuffle=shuffle)
        for inputs, targets in dataloader_viz:
            break
        single_feature_plt(inputs, targets, save_fig=viz_file)

    def train_neural_ode(self, hidden_dim=32, non_linearity='relu', lr=1e-3, num_epochs=12, visualize_features=True):
        model = ODENet(device, data_dim, hidden_dim, time_dependent=True,
                       non_linearity=non_linearity)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Set up trainer
        trainer = Trainer(model, optimizer, self.device)
        for inputs, targets in self.data_loader:
            break
        if visualize_features:
            feature_history = get_feature_history(trainer, self.data_loader, inputs,
                                                  targets, num_epochs)
        else:
            # If we don't record feature evolution, simply train model
            trainer.train(self.data_loader, num_epochs)
        multi_feature_plt(feature_history[::2], targets, save_fig='feature_history.png')


if __name__ == '__main__':
    # Config
    data_dim = 2
    device = torch.device('cpu')
    #########
    experiment_sphere = ExperimentSphere(data_dim=data_dim,device=device)
    experiment_sphere.load_data()
    experiment_sphere.viz_data()
    experiment_sphere.train_neural_ode()
    ########
