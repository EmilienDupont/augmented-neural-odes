import json
import matplotlib

matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import numpy as np
import os
import time
import torch
from anode.discrete_models import ResNet
from anode.models import ODENet
from anode.training import Trainer
from experiments.dataloaders import ConcentricSphere, ShiftedSines
from torch.utils.data import DataLoader
from viz.plots import histories_plt, multi_feature_plt


def run_experiments(device, data_dim=2, viz_batch_size=512, num_reps=5,
                    datasets=[], model_configs=[], training_config={}):
    """Runs experiments for various model configurations on various datasets.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    viz_batch_size : int
        Batch size (number of points) used for visualizations of learned
        features.

    num_reps : int
        Number of times to repeat experiments across different initializations.

    datasets : list of dicts
        List of dataset specifications, where each specification is a dict
        listing the properties of the data. Experiments will be run on each of
        these datasets.

    model_config : list of dicts
        List of model configurations on which to run experiments.

    training_config : dict
        Specifies training configurations

    Note
    ----
    For an example of the structure of the config files, see the config.json
    file.
    """
    results = []
    for dataset in datasets:
        if dataset["type"] == "sphere":
            data_object = ConcentricSphere(data_dim,
                                           dataset["inner_range"],
                                           dataset["outer_range"],
                                           dataset["num_points_inner"],
                                           dataset["num_points_outer"])
        elif dataset["type"] == "sine":
            data_object = ShiftedSines(data_dim,
                                       dataset["shift"],
                                       dataset["num_points_lower"],
                                       dataset["num_points_upper"],
                                       dataset["noise_scale"])

        data_loader = DataLoader(data_object,
                                 batch_size=training_config["batch_size"],
                                 shuffle=True)

        results.append({"dataset": dataset, "model_info": [], "tensors": [],
                        "models": []})

        # Retrieve inputs and targets which will be used to visualize how models
        # transform inputs to features
        data_loader_viz = DataLoader(data_object,
                                     batch_size=viz_batch_size,
                                     shuffle=True)
        for batch in data_loader_viz:
            break
        inputs, targets = batch

        for i, model_config in enumerate(model_configs):
            # Check whether model is ODE based or a ResNet
            is_ode = model_config["type"] == "odenet" or model_config["type"] == "anode"

            # Initialize histories
            loss_histories = []
            epoch_loss_histories = []
            epoch_nfe_histories = []
            features = []
            predictions = []
            models = []
            start = time.time()

            for j in range(num_reps):
                print("{}/{} model, {}/{} rep".format(i + 1, len(model_configs), j + 1, num_reps))

                if is_ode:
                    if model_config["type"] == "odenet":
                        augment_dim = 0
                    else:
                        augment_dim = model_config["augment_dim"]

                    model = ODENet(device, data_dim, model_config["hidden_dim"],
                                   augment_dim=augment_dim,
                                   time_dependent=model_config["time_dependent"])
                else:
                    model = ResNet(data_dim, model_config["hidden_dim"],
                                   model_config["num_layers"])

                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=model_config["lr"])

                trainer = Trainer(model, optimizer, device,
                                  print_freq=training_config["print_freq"],
                                  record_freq=training_config["record_freq"],
                                  verbose=False)

                trainer.train(data_loader, training_config["epochs"])

                loss_histories.append(trainer.histories["loss_history"])
                epoch_loss_histories.append(trainer.histories["epoch_loss_history"])
                if is_ode:
                    epoch_nfe_histories.append(trainer.histories["epoch_nfe_history"])

                # Add trained model
                models.append(model)

                # Record how model transforms inputs to features (this should
                # only happen when data_dim = 2)
                if data_dim == 2:
                    feats, preds = model(inputs, True)
                    features.append(feats.detach().cpu())
                    predictions.append(preds.detach().cpu())

            results[-1]["model_info"].append({
                "type": model_config["type"],
                "loss_history": loss_histories,
                "epoch_loss_history": epoch_loss_histories,
                "avg_time": (time.time() - start) / num_reps
            })

            if is_ode:
                results[-1]["model_info"][-1]["epoch_nfe_history"] = epoch_nfe_histories

            if data_dim == 2:
                results[-1]["tensors"].append({
                    "inputs": inputs.cpu(),
                    "targets": targets.cpu(),
                    "features": features,
                    "predictions": predictions
                })

            results[-1]["models"].append(models)

    return results


def run_experiments_from_config(device, path_to_config):
    """Runs an experiment from a config file.

    Parameters
    ----------
    device : torch.device

    path_to_config : string
    """
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    results = run_experiments(device, data_dim=config["data_dim"],
                              viz_batch_size=config["viz_batch_size"],
                              num_reps=config["num_reps"],
                              datasets=config["datasets"],
                              model_configs=config["model_configs"],
                              training_config=config["training_config"])

    return results


def run_and_save_experiments(device, path_to_config, save_models=False,
                             save_tensors=False):
    """Runs an experiment from a config file, saves logs and generates various
    plots of results.

    Parameters
    ----------
    device : torch.device

    path_to_config : string
        Path to config file.

    save_models : bool
        If True saves all models trained during experiments.

    save_tensors : bool
        If True saves input and feature tensors used to produce figures.
    """
    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "results_{}".format(timestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as config_file:
        json.dump(config, config_file)

    # Run experiments
    results = run_experiments_from_config(device, path_to_config)

    # Create figures and save experiments
    for i in range(len(results)):
        # Create directory to store result
        subdir = directory + '/{}'.format(i)
        os.makedirs(subdir)
        # Save dataset information
        with open(subdir + '/dataset.json', 'w') as f:
            json.dump(results[i]['dataset'], f)

        # Save model and losses info
        with open(subdir + '/model_losses.json', 'w') as f:
            json.dump(results[i]['model_info'], f)

        # Create losses figure for this dataset
        histories_plt(results[i]["model_info"], plot_type='loss',
                      save_fig=subdir + '/losses.png')
        histories_plt(results[i]["model_info"], plot_type='loss',
                      shaded_err=True, save_fig=subdir + '/losses_shaded.png')
        # Create number of function evaluations plots if ODE model is included
        contains_ode = False
        for model_config in config["model_configs"]:
            if model_config["type"] == "odenet" or model_config["type"] == "anode":
                contains_ode = True
                break
        if contains_ode:
            histories_plt(results[i]["model_info"], plot_type='nfe',
                          save_fig=subdir + '/nfes.png')
            histories_plt(results[i]["model_info"], plot_type='nfe',
                          shaded_err=True, save_fig=subdir + '/nfes_shaded.png')
            histories_plt(results[i]["model_info"], plot_type='nfe_vs_loss',
                          save_fig=subdir + '/nfe_vs_loss.png')

        # For each individual run, save model and save input-feature figures
        for j in range(len(results[i]["model_info"])):
            model_type = results[i]["model_info"][j]["type"]
            models = results[i]["models"][j]

            if save_models:
                for k in range(len(models)):
                    torch.save(models[k], subdir + '/model_{}_{}_{}.pt'.format(model_type, j, k))

            if save_tensors:
                # If data_dim is 2, extract the tensors and save them and make the
                # required plots
                if config["data_dim"] == 2:
                    tensors = results[i]["tensors"][j]
                    inputs = tensors["inputs"]
                    targets = tensors["targets"]
                    predictions = tensors["predictions"]

                    # Save tensors
                    torch.save(inputs, subdir + '/inputs_{}.pt'.format(j))
                    torch.save(targets, subdir + '/targets_{}.pt'.format(j))
                    torch.save(predictions, subdir + '/predictions_{}.pt'.format(j))

                    for k in range(len(tensors["features"])):
                        features = tensors["features"][k]
                        torch.save(features, subdir + '/features_{}.pt'.format(j))
                        # Create figure of inputs to features
                        multi_feature_plt([inputs, features], targets,
                                          save_fig=subdir + '/inp_to_feat_{}_{}_{}.png'.format(model_type, j, k))
