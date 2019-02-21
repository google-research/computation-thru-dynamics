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

""" Run the JAX based Integrator RNN training tutorial."""

from __future__ import print_function, division, absolute_import
import datetime
import h5py
import integrator
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacrev, jacfwd
from jax.experimental import optimizers
import jax.experimental.stax as stax
import jax.flatten_util as flatten_util
import matplotlib.pyplot as plt
import numpy as onp             # original CPU-backed NumPy
import os
import rnn as rnn
import time
import utils


### HYPERPARAMETERS

# Integration parameters
T = 1.0
ntimesteps = 25
bval = 0.01      # bias value limit
sval = 0.025     # standard deviation (before dividing by sqrt(dt))
input_params = (bval, sval, T, ntimesteps)

# Integrator RNN hyperparameters
u = 1
n = 100
o = 1
param_scale = 0.95

# Optimization hyperparameters
num_batchs = 3000
batch_size = 128
eval_batch_size = 1024
step_size = 0.02
decay_factor = 0.9995
decay_steps = 1
max_grad_norm = 10.0
l2reg = 0.00001
adam_b1 = 0.9
adam_b2 = 0.999
adam_eps = 1e-1
print_every = 50

seed = onp.random.randint(0, 1000000)
print("Seed: %d" % seed)
key = random.PRNGKey(seed)

# Plot a few input/target examples to make sure things look sane.
do_plot = False
if do_plot:
  ntoplot = 10
  key, subkey = random.split(key, 2)
  skeys = random.split(subkey, ntoplot)
  inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)
  plot_batch(ntimesteps, inputs, targets)


### TRAINING

# Init some parameters for training.
key, subkey = random.split(key, 2)
init_params = rnn.random_vrnn_params(subkey, u, n, o, g=param_scale)
decay_fun = optimizers.exponential_decay(step_size, decay_steps, decay_factor)
opt_init, opt_update = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(init_params)
# Run the optimization loop, first jit'd calls will take a minute.
start_time = time.time()
for batch in range(num_batchs):
  key, subkey = random.split(key, 2)
  skeys = random.split(subkey, batch_size)
  inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)
  opt_state = rnn.update_w_gc_jit(batch, opt_state, opt_update, inputs,
                                  targets, max_grad_norm, l2reg)
  if batch % print_every == 0:
    params = optimizers.get_params(opt_state)
    train_loss = rnn.loss_jit(params, inputs, targets, l2reg)
    batch_time = time.time() - start_time
    step_size = decay_fun(batch)
    s = "Batch {} in {:0.2f} sec, step size: {:0.5f}, training loss {:0.4f}"
    print(s.format(batch, batch_time, step_size, train_loss))
    start_time = time.time()


### TESTING    
# Take a batch for an evalulation loss
key, subkey = random.split(key, 2)
skeys = random.split(subkey, batch_size)
inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)
eval_loss = rnn.loss_jit(params, inputs, targets, l2reg=0.0)
eval_loss_str = "{:.5f}".format(eval_loss)
print("Loss on a new large batch: %s" % (eval_loss_str))


### SAVING 

# Save the parameters
data_dir = '/tmp/'
task_type = 'pure_int'
rnn_type = 'vrnn'

fname_uniquifier = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
params_fname = ('trained_params_' + rnn_type + '_' + task_type + '_' + \
                eval_loss_str + '_' + fname_uniquifier + '.h5')
params_fname = os.path.join(data_dir, params_fname)

print("Saving params in %s" % (params_fname))
utils.write_file(params_fname, params)

# Save about 10,000 trials for playing around with the LFADS tutorial.
data_fname = ('trained_data_' + rnn_type + '_' + task_type + '_' + \
              eval_loss_str + '_' + fname_uniquifier + '.h5')
data_fname = os.path.join(data_dir, data_fname)


# Save about 10000 trials
nsave_batches = 10
inputs_and_targets = \
    lambda keys: integrator.build_inputs_and_targets_jit(input_params, keys)
rnn_run = lambda inputs: rnn.batched_rnn_run(params, inputs)
data_dict = rnn.run_trials(rnn_run, inputs_and_targets, nsave_batches,
                           eval_batch_size)
print("Saving data in %s" %(data_fname))
utils.write_file(data_fname, data_dict)
