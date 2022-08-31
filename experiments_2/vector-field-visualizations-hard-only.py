#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import torch

device = torch.device('cpu')

# # ODE vector field visualizations
# This notebook shows examples of functions Neural ODEs cannot
# approximate and how this affects the learned vector fields.

# #### Create an ODE function

# In[2]:


from anode.models import ODEFunc

data_dim = 1  # We model 1d data to easily visualize it
hidden_dim = 16

# Create a 3-layer MLP as the ODE function f(h, t)
odefunc = ODEFunc(device, data_dim, hidden_dim, time_dependent=True)

# #### Visualize vector field of ODE function
# We can visualize what the randomly initialized ODE function's vector field looks like.

# In[3]:


from viz.plots import vector_field_plt

# vector_field_plt(odefunc, num_points=10, timesteps=10,h_min=-1.5, h_max=1.5,save_fig='./fig1.png')

# ## Create functions to approximate
# 
# We will approximate two functions: an easy one (the identity mapping) and a hard one (correspond to g_1d in the paper)

# In[4]:


from experiments.dataloaders import Data1D
from torch.utils.data import DataLoader

data_hard = Data1D(num_points=500, target_flip=True)

dataloader_hard = DataLoader(data_hard, batch_size=32, shuffle=True)

# #### Visualize the data

for inputs, targets in dataloader_hard:
    break


# ## Train a model on data
# 
# We can now try to fit a Neural ODE to the two functions

# In[7]:


from anode.models import ODEBlock
from anode.training import Trainer

data_dim = 1
hidden_dim = 16
# Create a model for the hard function
odefunc_hard = ODEFunc(device, data_dim, hidden_dim,
                       time_dependent=True)
model_hard = ODEBlock(device, odefunc_hard)


# Create an optimizer and trainer for hard function
optimizer_hard = torch.optim.Adam(model_hard.parameters(), lr=5e-4)
trainer_hard = Trainer(model_hard, optimizer_hard, device, print_freq=5)

# #### Train model on hard data

# In[10]:


trainer_hard.train(dataloader_hard, num_epochs=50)

# In[11]:


for inputs, targets in dataloader_hard:
    break

# Plot 8 trajectories
vector_field_plt(odefunc_hard, num_points=10, timesteps=10,
                 inputs=inputs[:8], targets=targets[:8],
                 h_min=-1.5, h_max=1.5, model=model_hard,save_fig='./hard_trajectory.png')
y = model_hard(inputs)
print(f"y_avg :{torch.nanmean(y)}")
print(f"y_std : {torch.std(y)}")
# ## Augmented Neural ODEs
# As can be seen, Neural ODEs struggle to fit the hard function.
# In fact, it can be proven that Neural ODEs cannot represent this function.
# In order to overcome this, we can use Augmented Neural ODEs which extend the space on which the ODE is solved.
# Examples of this are shown in the `augmented-neural-ode-example` notebook.
