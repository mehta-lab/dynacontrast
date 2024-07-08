# %% Imports and paths.
import os
import numpy as np
import argparse
import dask
import torch as t
import torch.nn as nn
import sys
from numcodecs import blosc
import torchview
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from train.resnet import EncodeProject
from train.losses import AllTripletMiner, NTXent
from utils.config_reader import YamlReader

dask.config.set(scheduler='synchronous')
blosc.use_threads = True

# %% Initialize the model
# read and apply model config
config_path = '../examples/config_example.yml'
config = YamlReader()
config.read_config(config_path)

network = config.training.network
network_width = config.training.network_width
num_inputs = config.training.num_inputs
loss = config.training.loss
temperature = config.training.temperature
device = 'cpu'

if loss == 'triplet':
    loss_fn = AllTripletMiner(margin=margin).to(device)
elif loss == 'ntxent':
    loss_fn = NTXent(tau=temperature).to(device)
else:
    raise ValueError('Loss name {} is not defined.'.format(loss))

# create model
model = EncodeProject(arch=network, loss=loss_fn, num_inputs=num_inputs, width=network_width, verbose = False,
                        viewmode = True).to(device)
print(model)

# %% Initialize the model
model_graph = torchview.draw_graph(
model,
t.randn(1, 2, 128, 128),
depth=3,  # adjust depth to zoom in.
device="cpu",
save_graph=True,
filename="model_view",
)

# Print the image of the model.
model_graph.resize_graph(scale=2.5)
model_graph.visual_graph
# %%
