import json
import matplotlib
matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import numpy as np
import os
import time
import torch
from anode.models import ODENet
from anode.conv_models import ConvODENet
from anode.discrete_models import ResNet
from anode.training import Trainer
from experiments.dataloaders import mnist, cifar10, tiny_imagenet
from viz.plots import histories_plt


def run_and_save_experiments_img(device, path_to_config):
    """Runs and saves experiments as they are produced (so results are still
    saved even if NFEs become excessively large or underflow occurs).

    Parameters
    ----------
    device : torch.device

    path_to_config : string
        Path to config json file.
    """
    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "img_results_{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as config_file:
        json.dump(config, config_file)

    num_reps = config["num_reps"]
    dataset = config["dataset"]
    model_configs = config["model_configs"]
    training_config = config["training_config"]

    results = {"dataset": dataset, "model_info": []}

    if dataset == 'mnist':
        data_loader, test_loader = mnist(training_config["batch_size"])
        img_size = (1, 28, 28)
        output_dim = 10

    if dataset == 'cifar10':
        data_loader, test_loader = cifar10(training_config["batch_size"])
        img_size = (3, 32, 32)
        output_dim = 10

    if dataset == 'imagenet':
        data_loader = tiny_imagenet(training_config["batch_size"])
        img_size = (3, 64, 64)
        output_dim = 200

    only_success = True  # Boolean to keep track of any experiments failing

    for i, model_config in enumerate(model_configs):
        results["model_info"].append({})
        # Keep track of losses and nfes
        loss_histories = []
        nfe_histories = []
        bnfe_histories = []
        total_nfe_histories = []
        epoch_loss_histories = []
        epoch_nfe_histories = []
        epoch_bnfe_histories = []
        epoch_total_nfe_histories = []
        # Keep track of models potentially failing
        model_stats = {
            "exceeded": {"count": 0, "final_losses": [], "final_nfes": [],
                         "final_bnfes": []},
            "underflow": {"count": 0, "final_losses": [], "final_nfes": [],
                          "final_bnfes": []},
            "success": {"count": 0, "final_losses": [], "final_nfes": [],
                        "final_bnfes": []}
        }

        if model_config["validation"]:
            epoch_loss_val_histories = []

        is_ode = model_config["type"] == "odenet" or model_config["type"] == "anode"

        for j in range(num_reps):
            print("{}/{} model, {}/{} rep".format(i + 1, len(model_configs), j + 1, num_reps))

            if is_ode:
                if model_config["type"] == "odenet":
                    augment_dim = 0
                else:
                    augment_dim = model_config["augment_dim"]

                model = ConvODENet(device, img_size, model_config["num_filters"],
                                   output_dim=output_dim,
                                   augment_dim=augment_dim,
                                   time_dependent=model_config["time_dependent"],
                                   non_linearity=model_config["non_linearity"],
                                   adjoint=True)
            else:
                model = ResNet(data_dim, model_config["hidden_dim"],
                               model_config["num_layers"],
                               output_dim=output_dim,
                               is_img=True)

            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=model_config["lr"],
                                         weight_decay=model_config["weight_decay"])

            trainer = Trainer(model, optimizer, device,
                              classification=True,
                              print_freq=training_config["print_freq"],
                              record_freq=training_config["record_freq"],
                              verbose=True,
                              save_dir=(directory, '{}_{}'.format(i, j)))

            loss_histories.append([])
            epoch_loss_histories.append([])
            nfe_histories.append([])
            epoch_nfe_histories.append([])
            bnfe_histories.append([])
            epoch_bnfe_histories.append([])
            total_nfe_histories.append([])
            epoch_total_nfe_histories.append([])

            if model_config["validation"]:
                epoch_loss_val_histories.append([])

            # Train one epoch at a time, as NODEs can underflow or exceed the
            # maximum NFEs
            for epoch in range(training_config["epochs"]):
                print("\nEpoch {}".format(epoch + 1))
                try:
                    trainer.train(data_loader, 1)
                    end_training = False
                except AssertionError as e:
                    only_success = False
                    # Assertion error means we either underflowed or exceeded
                    # the maximum number of steps
                    error_message = e.args[0]
                    # Error message in torchdiffeq for max_num_steps starts
                    # with 'max_num_steps'
                    if error_message.startswith("max_num_steps"):
                        print("Maximum number of steps exceeded")
                        file_name_root = 'exceeded'
                    elif error_message.startswith("underflow"):
                        print("Underflow")
                        file_name_root = 'underflow'
                    else:
                        print("Unknown assertion error")
                        file_name_root = 'unknown'

                    model_stats[file_name_root]["count"] += 1

                    if len(trainer.buffer['loss']):
                        final_loss = np.mean(trainer.buffer['loss'])
                    else:
                        final_loss = None
                    model_stats[file_name_root]["final_losses"].append(final_loss)

                    if len(trainer.buffer['nfe']):
                        final_nfes = np.mean(trainer.buffer['nfe'])
                    else:
                        final_nfes = None
                    model_stats[file_name_root]["final_nfes"].append(final_nfes)

                    if len(trainer.buffer['bnfe']):
                        final_bnfes = np.mean(trainer.buffer['bnfe'])
                    else:
                        final_bnfes = None
                    model_stats[file_name_root]["final_bnfes"].append(final_bnfes)

                    # Save final NFEs before error happened
                    with open(directory + '/{}_{}_{}.json'.format(file_name_root, i, j), 'w') as f:
                        json.dump({"forward": trainer.nfe_buffer, "backward": trainer.bnfe_buffer}, f)

                    end_training = True

                # Save info at every epoch
                loss_histories[-1] = trainer.histories['loss_history']
                epoch_loss_histories[-1] = trainer.histories['epoch_loss_history']
                if is_ode:
                    nfe_histories[-1] = trainer.histories['nfe_history']
                    epoch_nfe_histories[-1] = trainer.histories['epoch_nfe_history']
                    bnfe_histories[-1] = trainer.histories['bnfe_history']
                    epoch_bnfe_histories[-1] = trainer.histories['epoch_bnfe_history']
                    total_nfe_histories[-1] = trainer.histories['total_nfe_history']
                    epoch_total_nfe_histories[-1] = trainer.histories['epoch_total_nfe_history']

                if model_config["validation"]:
                    epoch_loss_val = dataset_mean_loss(trainer, test_loader, device)
                    if epoch == 0:
                        epoch_loss_val_histories[-1] = [epoch_loss_val]
                    else:
                        epoch_loss_val_histories[-1].append(epoch_loss_val)

                results["model_info"][-1]["type"] = model_config["type"]
                results["model_info"][-1]["loss_history"] = loss_histories
                results["model_info"][-1]["epoch_loss_history"] = epoch_loss_histories
                if model_config["validation"]:
                    results["model_info"][-1]["epoch_loss_val_history"] = epoch_loss_val_histories

                if is_ode:
                    results["model_info"][-1]["epoch_nfe_history"] = epoch_nfe_histories
                    results["model_info"][-1]["nfe_history"] = nfe_histories
                    results["model_info"][-1]["epoch_bnfe_history"] = epoch_bnfe_histories
                    results["model_info"][-1]["bnfe_history"] = bnfe_histories
                    results["model_info"][-1]["epoch_total_nfe_history"] = epoch_total_nfe_histories
                    results["model_info"][-1]["total_nfe_history"] = total_nfe_histories

                # Save losses and nfes at every epoch
                with open(directory + '/losses_and_nfes.json', 'w') as f:
                    json.dump(results['model_info'], f)

                # If training failed, move on to next rep
                if end_training:
                    break

                # If we reached end of training, increment success counter
                if epoch == training_config["epochs"] - 1:
                    model_stats["success"]["count"] += 1

                    if len(trainer.buffer['loss']):
                        final_loss = np.mean(trainer.buffer['loss'])
                    else:
                        final_loss = None
                    model_stats["success"]["final_losses"].append(final_loss)

                    if len(trainer.buffer['nfe']):
                        final_nfes = np.mean(trainer.buffer['nfe'])
                    else:
                        final_nfes = None
                    model_stats["success"]["final_nfes"].append(final_nfes)

                    if len(trainer.buffer['bnfe']):
                        final_bnfes = np.mean(trainer.buffer['bnfe'])
                    else:
                        final_bnfes = None
                    model_stats["success"]["final_bnfes"].append(final_bnfes)

        # Save model stats
        with open(directory + '/model_stats{}.json'.format(i), 'w') as f:
            json.dump(model_stats, f)

    # Create plots

    # Extract size of augmented dims
    augment_labels = ['p = 0' if model_config['type'] == 'odenet' else 'p = {}'.format(model_config['augment_dim'])
                      for model_config in config['model_configs']]
    # Create losses figure
    # Note that we can only calculate mean loss if all models trained to
    # completion. Therefore we only include mean if only_success is True
    histories_plt(results["model_info"], plot_type='loss', labels=augment_labels,
                  include_mean=only_success, save_fig=directory + '/losses.png')
    histories_plt(results["model_info"], plot_type='loss', labels=augment_labels,
                  include_mean=only_success, shaded_err=True, save_fig=directory + '/losses_shaded.png')

    # Create NFE plots if ODE model is included
    contains_ode = False
    for model_config in config["model_configs"]:
        if model_config["type"] == "odenet" or model_config["type"] == "anode":
            contains_ode = True
            break

    if contains_ode:
        # If adjoint method was used, plot forwards, backwards and total nfes
        if trainer.model.odeblock.adjoint:
            nfe_types = ['nfe', 'bnfe', 'total_nfe']
        else:
            nfe_types = ['nfe']

        for nfe_type in nfe_types:
            histories_plt(results["model_info"], plot_type='nfe', labels=augment_labels,
                          include_mean=only_success, nfe_type=nfe_type,
                          save_fig=directory + '/{}s.png'.format(nfe_type))
            histories_plt(results["model_info"], plot_type='nfe', labels=augment_labels,
                          include_mean=only_success, shaded_err=True, nfe_type=nfe_type,
                          save_fig=directory + '/{}s_shaded.png'.format(nfe_type))
            histories_plt(results["model_info"], plot_type='nfe_vs_loss', labels=augment_labels,
                          include_mean=only_success, nfe_type=nfe_type,
                          save_fig=directory + '/{}_vs_loss.png'.format(nfe_type))


def dataset_mean_loss(trainer, data_loader, device):
    """Returns mean loss of model on a dataset. Useful for calculating
    validation loss.

    Parameters
    ----------
    trainer : training.Trainer instance
        Trainer instance for model we want to evaluate.

    data_loader : torch.utils.data.DataLoader

    device : torch.device
    """
    epoch_loss = 0.
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = trainer.model(x_batch)
        loss = trainer._loss(y_pred, y_batch)
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
