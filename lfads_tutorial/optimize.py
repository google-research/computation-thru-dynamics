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


"""Optimization routines for LFADS"""


from __future__ import print_function, division, absolute_import

import datetime
import h5py

import jax.numpy as np
from jax import grad, jit, lax, random
from jax.experimental import optimizers
import jax.flatten_util as flatten_util

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

import lfads_tutorial.lfads as lfads
import lfads_tutorial.utils as utils

import time


def get_kl_warmup_fun(lfads_opt_hps):
  """Warmup KL cost to avoid a pathological condition early in training.

  Arguments:
    lfads_opt_hps : dictionary of optimization hyperparameters

  Returns:
    a function which yields the warmup value
  """

  kl_warmup_start = lfads_opt_hps['kl_warmup_start']
  kl_warmup_end = lfads_opt_hps['kl_warmup_end']
  kl_min = lfads_opt_hps['kl_min']
  kl_max = lfads_opt_hps['kl_max']
  def kl_warmup(batch_idx):
    progress_frac = ((batch_idx - kl_warmup_start) /
                     (kl_warmup_end - kl_warmup_start))
    kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                         (kl_max - kl_min) * progress_frac + kl_min)
    return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
  return kl_warmup


def get_update_w_gc_fun(init_params, opt_update):
  """Update the parameters w/ gradient clipped, gradient descent updates.

  Arguments:
    init_params: parameter dictionary
    opt_update: a function that updates the parameters (from jax.optimizers)

  Returns:
    a function which updates the parameters according to the optimizer.
  """
  unflatten_lfads = flatten_util.ravel_pytree(init_params)[1]
  flatten_lfads = lambda params: flatten_util.ravel_pytree(params)[0]

  def update_w_gc(i, opt_state, lfads_hps, lfads_opt_hps, key, x_bxt,
                  kl_warmup):
    max_grad_norm = lfads_opt_hps['max_grad_norm']
    keep_rate = lfads_opt_hps['keep_rate']

    params = optimizers.get_params(opt_state)

    grads = grad(lfads.lfads_training_loss)(params, lfads_hps, key, x_bxt,
                                            kl_warmup, keep_rate)
    flat_grads = flatten_lfads(grads)
    grad_norm = np.sqrt(np.sum(flat_grads**2))
    normed_grads = np.where(grad_norm <= max_grad_norm, flat_grads,
                            flat_grads * (max_grad_norm / grad_norm))
    uf_grads = unflatten_lfads(normed_grads)
    return opt_update(i, uf_grads, opt_state)

  return update_w_gc


def optimize_lfads_core(key, batch_idx_start, num_batches,
                        update_fun, kl_warmup_fun,
                        opt_state, lfads_hps, lfads_opt_hps, train_data):
  """Make gradient updates to the LFADS model.

  Uses lax.fori_loop instead of a Python loop to reduce JAX overhead. This 
    loop will be jit'd and run on device.

  Arguments:
    init_params: a dict of parameters to be trained
    batch_idx_start: Where are we in the total number of batches
    num_batches: how many batches to run
    update_fun: the function that changes params based on grad of loss
    kl_warmup_fun: function to compute the kl warmup
    opt_state: the jax optimizer state, containing params and opt state
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  key, dkeyg = utils.keygen(key, num_batches) # data
  key, fkeyg = utils.keygen(key, num_batches) # forward pass
  
  # Begin optimziation loop. Explicitly avoiding a python for-loop
  # so that jax will not trace it for the sake of a gradient we will not use.  
  def run_update(batch_idx, opt_state):
    kl_warmup = kl_warmup_fun(batch_idx)
    didxs = random.randint(next(dkeyg), [lfads_hps['batch_size']], 0,
                           train_data.shape[0])
    x_bxt = train_data[didxs].astype(np.float32)
    opt_state = update_fun(batch_idx, opt_state, lfads_hps, lfads_opt_hps,
                           next(fkeyg), x_bxt, kl_warmup)
    return opt_state

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  return lax.fori_loop(lower, upper, run_update, opt_state)


optimize_lfads_core_jit = jit(optimize_lfads_core, static_argnums=(2,3,4,6,7))


def optimize_lfads(key, init_params, lfads_hps, lfads_opt_hps,
                   train_data, eval_data):
  """Optimize the LFADS model and print batch based optimization data.

  This loop is at the cpu nonjax-numpy level.

  Arguments:
    init_params: a dict of parameters to be trained
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    a dictionary of trained parameters"""
  
  # Begin optimziation loop.
  all_tlosses = []
  all_elosses = []

  # Build some functions used in optimization.
  kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
  decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],
                                           lfads_opt_hps['decay_steps'],
                                           lfads_opt_hps['decay_factor'])

  opt_init, opt_update = optimizers.adam(step_size=decay_fun,
                                         b1=lfads_opt_hps['adam_b1'],
                                         b2=lfads_opt_hps['adam_b2'],
                                         eps=lfads_opt_hps['adam_eps'])
  opt_state = opt_init(init_params)
  update_fun = get_update_w_gc_fun(init_params, opt_update)

  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = lfads_hps['batch_size']
  num_batches = lfads_opt_hps['num_batches']
  print_every = lfads_opt_hps['print_every']
  num_opt_loops = int(num_batches / print_every)
  params = optimizers.get_params(opt_state)
  for oidx in range(num_opt_loops):
    batch_idx_start = oidx * print_every
    start_time = time.time()
    key, tkey, dtkey, dekey = random.split(random.fold_in(key, oidx), 4)
    opt_state = optimize_lfads_core_jit(tkey, batch_idx_start,
                                        print_every, update_fun, kl_warmup_fun,
                                        opt_state, lfads_hps, lfads_opt_hps,
                                        train_data)
    batch_time = time.time() - start_time

    # Losses
    params = optimizers.get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    kl_warmup = kl_warmup_fun(batch_idx_start)
    # Training loss
    didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    x_bxt = train_data[didxs].astype(onp.float32)
    tlosses = lfads.lfads_losses_jit(params, lfads_hps, dtkey, x_bxt,
                                     kl_warmup, 1.0)

    # Evaluation loss
    didxs = onp.random.randint(0, eval_data.shape[0], batch_size)
    ex_bxt = eval_data[didxs].astype(onp.float32)
    elosses = lfads.lfads_losses_jit(params, lfads_hps, dekey, ex_bxt,
                                     kl_warmup, 1.0)
    # Saving, printing.
    all_tlosses.append(tlosses)
    all_elosses.append(elosses)
    s = "Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}, Training loss {:0.0f}, Eval loss {:0.0f}"
    print(s.format(batch_idx_start+1, batch_pidx, batch_time,
                   decay_fun(batch_pidx), tlosses['total'], elosses['total']))

    tlosses_thru_training = utils.merge_losses_dicts(all_tlosses)
    elosses_thru_training = utils.merge_losses_dicts(all_elosses)
    optimizer_details = {'tlosses' : tlosses_thru_training,
                         'elosses' : elosses_thru_training}
    
  return params, optimizer_details


  
