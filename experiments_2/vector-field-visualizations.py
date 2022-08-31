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

data_easy = Data1D(num_points=500, target_flip=False)
data_hard = Data1D(num_points=500, target_flip=True)

dataloader_easy = DataLoader(data_easy, batch_size=32, shuffle=True)
dataloader_hard = DataLoader(data_hard, batch_size=32, shuffle=True)

# #### Visualize the data

# In[5]:


for inputs, targets in dataloader_easy:
    break

vector_field_plt(odefunc, num_points=10, timesteps=10,
                 inputs=inputs, targets=targets,
                 h_min=-1.5, h_max=1.5,save_fig='easy_data.png')

# In[6]:


for inputs, targets in dataloader_hard:
    break

vector_field_plt(odefunc, num_points=10, timesteps=10,
                 inputs=inputs, targets=targets,
                 h_min=-1.5, h_max=1.5,save_fig='./hard_data.png')

# ## Train a model on data
# 
# We can now try to fit a Neural ODE to the two functions

# In[7]:


from anode.models import ODEBlock
from anode.training import Trainer

data_dim = 1
hidden_dim = 16

# Create a model for the easy function
odefunc_easy = ODEFunc(device, data_dim, hidden_dim,
                       time_dependent=True)
model_easy = ODEBlock(device, odefunc_easy)

# Create a model for the hard function
odefunc_hard = ODEFunc(device, data_dim, hidden_dim,
                       time_dependent=True)
model_hard = ODEBlock(device, odefunc_hard)

# Create an optimizer and trainer for easy function
optimizer_easy = torch.optim.Adam(model_easy.parameters(), lr=1e-3)
trainer_easy = Trainer(model_easy, optimizer_easy, device, print_freq=5)

# Create an optimizer and trainer for hard function
optimizer_hard = torch.optim.Adam(model_hard.parameters(), lr=5e-4)
trainer_hard = Trainer(model_hard, optimizer_hard, device, print_freq=5)

# #### Train model on easy data

# In[8]:


trainer_easy.train(dataloader_easy, num_epochs=10)

# #### Visualize model trajectories
# As can be seen, the learned vector field maps the inputs to targets correctly.

# In[9]:


for inputs, targets in dataloader_easy:
    break

# Plot 8 trajectories
vector_field_plt(odefunc_easy, num_points=10, timesteps=10,
                 inputs=inputs[:8], targets=targets[:8],
                 h_min=-1.5, h_max=1.5, model=model_easy,save_fig='./easy_trajectory.png')

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

