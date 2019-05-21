import sys
import torch
from experiments.experiments import run_and_save_experiments

device = torch.device('cpu')

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
config_path = sys.argv[1]

run_and_save_experiments(device, config_path)
