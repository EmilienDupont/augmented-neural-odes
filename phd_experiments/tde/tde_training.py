import logging
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class TensorDiffEqTrainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 train_data_loader: DataLoader, test_data_loader: DataLoader, device: torch.device,
                 loss_func: Union[torch.nn.SmoothL1Loss, torch.nn.CrossEntropyLoss], num_epochs: int = 20,
                 lr: float = 1e-3,
                 print_freq: int = 10):
        """

        Parameters
        ----------
        model
        optimizer
        train_data_loader
        device
        loss_func
        num_epochs
        lr
        print_freq
        """
        self.loss_func = loss_func
        self.device = device
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.logger = logging.getLogger()

    def _train_epoch(self, model: torch.nn.Module):
        epoch_tot_loss = 0
        for i, (x_batch, y_batch) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = model(x_batch)
            loss = self.loss_func(y_batch, y_pred)

            epoch_tot_loss += loss.item()

            loss.backward()
            self.optimizer.step()
        avg_epoch_loss = epoch_tot_loss / len(self.train_data_loader)
        return avg_epoch_loss

    def train(self, model: torch.nn.Module, verbose=False):
        epoch_losses_history = []
        self.logger.info(f'Epochs ranges from 1 to {self.num_epochs}')
        epoch_loss = np.Inf
        for epoch_idx in tqdm(range(self.num_epochs), desc='Epochs progress'):
            epoch_loss = self._train_epoch(model)
            if verbose and epoch_idx % self.print_freq == 0:
                self.logger.debug(f'Epoch # {epoch_idx} , epoch avg loss = {epoch_loss}')
            epoch_losses_history.append(epoch_loss)
        self.logger.info(f'Final epoch avg loss at epoch {self.num_epochs} = {epoch_loss}')
        return model, epoch_losses_history

    def evaluate(self, model: torch.nn.Module):
        batches_losses = []
        for i, (x_batch, y_batch) in enumerate(self.test_data_loader):
            y_pred = model(x_batch)
            batch_loss = self.loss_func(y_pred, y_batch)
            # FIXME for debugging
            # df = pd.DataFrame(
            #     {'y_pred': y_pred.view(-1).detach().numpy(), 'y_batch': y_batch.view(-1).detach().numpy()})
            batches_losses.append(batch_loss.item())
        avg_batches_loss = np.nanmean(np.array(batches_losses))
        return avg_batches_loss

    @staticmethod
    def diagnose(model: torch.nn.Module):
        # sparsity evolution
        P_sparsity_evol = [TensorDiffEqTrainer.measure_sparsity(x) for x in model.monitor['P']]
        U_sparsity_evol = [TensorDiffEqTrainer.measure_sparsity(x) for x in model.monitor['U']]
        F_sparsity_evol = [TensorDiffEqTrainer.measure_sparsity(x) for x in model.monitor['F']]

        # Norm evolution
        P_norm_evol = [torch.norm(x).item() for x in model.monitor['P']]
        U_norm_evol = [torch.norm(x).item() for x in model.monitor['U']]
        F_norm_evol = [torch.norm(x).item() for x in model.monitor['F']]

        # Diff for successive iterations
        P_diff = pd.Series(data=[x.detach().numpy() for x in model.monitor['P']]).diff().apply(
            func=lambda x: np.linalg.norm(x))
        U_diff = pd.Series(data=[x.detach().numpy() for x in model.monitor['U']]).diff().apply(
            func=lambda x: np.linalg.norm(x))
        F_diff = pd.Series(data=[x.detach().numpy() for x in model.monitor['F']]).diff().apply(
            func=lambda x: np.linalg.norm(x))
        print("")

    @staticmethod
    def measure_sparsity(tensor: torch.Tensor):
        t = tensor < 1e-5
        sparsity = torch.sum(t) / torch.numel(t)
        return sparsity.item()
