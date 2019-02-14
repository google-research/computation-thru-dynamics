# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run the LFADS algorithm on an RNN that integrates white noise."""


from __future__ import print_function, division, absolute_import

import datetime
import h5py

import jax.numpy as np
from jax import grad, jit, random, vmap
from jax import jacrev, jacfwd
from jax.experimental import optimizers
import jax.flatten_util as flatten_util

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn
import time

import lfads
from optimize import optimize_lfads
import os
import plotting
import utils


onp_rng = onp.random.RandomState(seed=0) # For here-level numpy


### LOAD THE DATA
integrator_rnn_data_file = \
     'trained_data_vrnn_pure_int_0.00735_2019-02-06_19_46_04.h5'
data_dt = 1.0/25.0
max_firing_rate = 20            # spikes per second

lfads_dir = '/tmp/lfads/'
data_dir = os.path.join(lfads_dir, 'data/')
output_dir = os.path.join(lfads_dir, 'output/')
figure_dir = os.path.join(lfads_dir, os.path.join(output_dir, 'figures/'))
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

data_path = os.path.join(data_dir, integrator_rnn_data_file)
data_dict = utils.read_file(data_path)
do_plot=False
if do_plot:
  plotting.plot_data_pca(data_dict)
  plt.savefig(os.path.join(figure_dir, 'data_pca.png'))
  plotting.plot_data_example(data_dict['inputs'], data_dict['hiddens'],
                             data_dict['outputs'], data_dict['targets'])
  plt.savefig(os.path.join(figure_dir, 'data_example.png'))

# Data was generated w/ VRNN w/ tanh, thus (data+1) / 2 -> [0,1]
data_bxtxn = utils.spikify_data((data_dict['hiddens'] + 1)/2, onp_rng, data_dt,
                                max_firing_rate=max_firing_rate)
if do_plot:
  plotting.plot_data_stats(data_dict, data_bxtxn, data_dt)
  plt.savefig(os.path.join(figure_dir, 'data_stats.png'))
train_data, eval_data = utils.split_data(data_bxtxn, train_fraction=0.9)


### LFADS Hyper parameters
data_dim = train_data.shape[2]  # input to lfads should have dimensions:
ntimesteps = train_data.shape[1] #   (batch_size x ntimesteps x data_dim)
batch_size = 128      # batch size during optimization

# LFADS architecture
enc_dim = 64          # encoder dim
con_dim = 64          # contoller dim
ii_dim = 1            # inferred input dim
gen_dim = 75          # generator dim
factors_dim = 20      # factors dim

# Optimization HPs that percolates into model
l2reg = 0.000002      # amount of l2 on weights

# Prior distribution parameters
ar_mean = 0.0
ar_autocorrelation_tau = 1.0
ar_noise_variance = 0.1
ic_prior_var = 0.1

lfads_hps = {'data_dim' : data_dim, 'ntimesteps' : ntimesteps,
             'enc_dim' : enc_dim, 'con_dim' : con_dim,
             'ic_prior_var' : ic_prior_var, 'ar_mean' : ar_mean,
             'ar_autocorrelation_tau' : ar_autocorrelation_tau,
             'ar_noise_variance' : ar_noise_variance,
             'ii_dim' : ii_dim, 'gen_dim' : gen_dim,
             'factors_dim' : factors_dim,
             'l2reg' : l2reg,
             'batch_size' : batch_size}


### LFADS Optimization hyperparameters
num_batches = 3000
print_every = 10
step_size = 0.05
decay_factor = 0.999
decay_steps = 1
keep_rate = 0.98                # dropout keep rate during training
max_grad_norm = 20.0            # gradient clipping above this value
kl_warmup_start = 500
kl_warmup_end = 1500
kl_max = 1.0
l2reg = 0.000002      # amount of l2 on weights (in lfads_hps)

lfads_opt_hps = {'num_batches' : num_batches, 'step_size' : step_size,
                 'decay_steps' : decay_steps, 'decay_factor' : decay_factor,
                 'kl_max' : kl_max, 'kl_warmup_start' : kl_warmup_start,
                 'kl_warmup_end' : kl_warmup_end, 'keep_rate' : keep_rate,
                 'max_grad_norm' : max_grad_norm, 'print_every' : print_every,
                 'adam_b1' : 0.9, 'adam_b2' : 0.999, 'adam_eps' : 1e-1}



### TRAINING LFADS MODEL

# Initialize parameters for LFADS and reset optimization loop counters.
key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
init_params = lfads.lfads_params(key, lfads_hps)

# Get the trained parameters and check out the losses.
trained_params, losses_dict = \
    optimize_lfads(init_params, lfads_hps, lfads_opt_hps, train_data, eval_data)

# Plot some information about the training.
if do_plot:
  plotting.plot_losses(losses_dict['tlosses'],
                       losses_dict['elosses'],
                       sampled_every=10, start_idx=100, stop_idx=num_batches)
  plt.savefig(os.path.join(figure_dir, 'losses.png'))


# Here, have to implement after LFADS is working again.
#nexamples_to_save = 10
#plotting.plot_lfads()

# Create a savename for the trained parameters and save them.
rnn_type = 'lfads'
task_type = 'integrator'
fname_uniquifier = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
network_fname = ('trained_params_' + rnn_type + '_' + task_type + '_' + \
                 fname_uniquifier + '.npz')
network_path = os.path.join(output_dir, network_fname)
print(network_path)
onp.savez(network_path, trained_params)
