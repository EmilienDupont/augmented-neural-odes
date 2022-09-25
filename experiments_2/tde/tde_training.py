import logging
from typing import Union

import torch
from torch.utils.data import DataLoader


class TensorDiffEqTrainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader, device: torch.device,
                 loss_func: Union[torch.nn.SmoothL1Loss, torch.nn.CrossEntropyLoss], num_epochs: int = 20,
                 lr: float = 1e-3,
                 print_freq: int = 10):
        """

        Parameters
        ----------
        model
        optimizer
        data_loader
        device
        loss_func
        num_epochs
        lr
        print_freq
        """
        self.loss_func = loss_func
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.logger = logging.getLogger()

    def _train_epoch(self, epoch_idx, verbose):
        epoch_tot_loss = 0
        for i, (x_batch, y_batch) in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = self.model(x_batch)
            loss = self.loss_func(y_batch, y_pred)
            if verbose:
                if i % self.print_freq == 0:
                    self.logger.debug(f'Epoch # {epoch_idx + 1} and batch-iter # {i + 1} : loss = {loss.item()}')

            epoch_tot_loss += loss.item()

            loss.backward()
            self.optimizer.step()
        avg_epoch_loss = epoch_tot_loss / len(self.data_loader)
        return avg_epoch_loss

    def train(self, verbose=False):
        epoch_losses_history = []
        for epoch_idx in range(self.num_epochs):
            epoch_loss = self._train_epoch(epoch_idx, verbose)
            epoch_losses_history.append(epoch_loss)
        return epoch_losses_history
