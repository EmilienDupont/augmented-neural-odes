# Augmented Neural ODEs

This repo contains code for the paper [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) (2019).

## Examples

<img src="https://github.com/EmilienDupont/augmented-neural-odes/raw/master/imgs/summary.gif" width="500">

## Requirements

The requirements that can be directly installed from PyPi can be found in `requirements.txt`. This code also builds on the awesome `torchdiffeq` library, which provides various ODE solvers on GPU. Instructions for installing `torchdiffeq` can be found in this [repo](https://github.com/rtqichen/torchdiffeq).

## Usage

The usage pattern is simple:

```python
# ... Load some data ...

import torch
from anode.conv_models import ConvODENet
from anode.models import ODENet
from anode.training import Trainer

# Instantiate a model
# For regular data...
anode = ODENet(device, data_dim=2, hidden_dim=16, augment_dim=1)
# ... or for images
anode = ConvODENet(device, img_size=(1, 28, 28), num_filters=32, augment_dim=1)

# Instantiate an optimizer and a trainer
optimizer = torch.optim.Adam(anode.parameters(), lr=1e-3)
trainer = Trainer(anode, optimizer, device)

# Train model on your dataloader
trainer.train(dataloader, num_epochs=10)
```

More detailed examples and tutorials can be found in the `augmented-neural-ode-example.ipynb` and `vector-field-visualizations.ipynb` notebooks.

### Running experiments

To run a large number of repeated experiments on toy datasets, use the following

```
python main_experiment.py config.json
```

where the specifications for the experiment can be found in `config.json`. This will log all the information about the experiments and generate plots for losses, NFEs and so on.

### Running experiments on image datasets

To run large experiments on image datasets, use the following

```
python main_experiment_img.py config_img.json
```

where the specifications for the experiment can be found in `config_img.json`.

## Demos

We also provide two demo notebooks that show how to reproduce some of the results and figures from the paper.

### Vector fields

<img src="https://github.com/EmilienDupont/augmented-neural-odes/raw/master/imgs/vector_field.png" width="500">

The `vector-field-visualizations.ipynb` notebook contains a demo and tutorial for reproducing the experiments on 1D ODE flows in the paper.

### Augmented Neural ODEs

<img src="https://github.com/EmilienDupont/augmented-neural-odes/raw/master/imgs/feature_history.png" width="800">

The `augmented-neural-ode-example.ipynb` notebook contains a demo and tutorial for reproducing the experiments comparing Neural ODEs and Augmented Neural ODEs on simple 2D functions.

## Data

The MNIST and CIFAR10 datasets can be directly downloaded using `torchvision` (this will happen automatically if you run the code, unless you already have those datasets downloaded). To run experiments on ImageNet, you will need to download the data from the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) website.

## Citing

If you find this code useful in your research, consider citing with

```
@article{dupont2019augmented,
  title={Augmented Neural ODEs},
  author={Dupont, Emilien and Doucet, Arnaud and Teh, Yee Whye},
  journal={arXiv preprint arXiv:1904.01681},
  year={2019}
}
```

## License

MIT
