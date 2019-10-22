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

from jax import grad, jit, lax, random
from jax.experimental import optimizers
from jax.nn import softmax
import jax.numpy as np

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

import lfads_gmm.lfads as lfads
import lfads_gmm.utils as utils

import time


def get_kl_warmup_fun(opt_hps):
  """Warmup KL cost to avoid a pathological condition early in training.

  Arguments:
    opt_hps : dictionary of optimization hyperparameters

  Returns:
    a function which yields the warmup value
  """

  kl_warmup_start = opt_hps['kl_warmup_start']
  kl_warmup_end = opt_hps['kl_warmup_end']
  kl_min = opt_hps['kl_min']
  kl_max = opt_hps['kl_max']
  def kl_warmup(batch_idx):
    progress_frac = ((batch_idx - kl_warmup_start) /
                     (kl_warmup_end - kl_warmup_start))
    kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                         (kl_max - kl_min) * progress_frac + kl_min)
    return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
  return kl_warmup


def optimize_core(key, batch_idx_start, num_batches,
                        update_fun, kl_warmup_fun,
                        opt_state, hps, opt_hps, train_data_fun):
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
    hps: dict of lfads model HPs
    opt_hps: dict of optimization HPs
    train_data_fun: key -> nexamples x time x ndims np array of data

  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  # Begin optimziation loop. Explicitly avoiding a python for-loop
  # so that jax will not trace it for the sake of a gradient we will not use.
  def run_update(batch_idx, opt_state_n_keys):
    opt_state, keys = opt_state_n_keys
    dkey, fkey = keys
    dkey = random.fold_in(dkey, batch_idx)
    fkey = random.fold_in(fkey, batch_idx)    
    kl_warmup = kl_warmup_fun(batch_idx)
    x_bxt = train_data_fun(dkey).astype(np.float32)
    opt_state = update_fun(batch_idx, opt_state, hps, opt_hps,
                           fkey, x_bxt, kl_warmup)
    opt_state_n_keys = (opt_state, (dkey, fkey))
    return opt_state_n_keys

  dkey, fkey = random.split(key, 2)
  opt_state_n_keys = (opt_state, (dkey, fkey))
  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  opt_state_n_keys = lax.fori_loop(lower, upper, run_update, opt_state_n_keys)
  opt_state, _ = opt_state_n_keys
  return opt_state


optimize_core_jit = jit(optimize_core, static_argnums=(2,3,4,6,7,8))


def optimize_lfads(key, init_params, hps, opt_hps,
                   train_data_fun, eval_data_fun):
  """Optimize the LFADS model and print batch based optimization data.

  This loop is at the cpu nonjax-numpy level.

  Arguments:
    init_params: a dict of parameters to be trained
    hps: dict of lfads model HPs
    opt_hps: dict of optimization HPs
    train_data_fun: function that takes a key and returns
      nexamples x time x ndims np array of data for training
    eval_data_fun: function that takes a key and returns
      nexamples x time x ndims np array of data for held out error
  Returns:
    a dictionary of trained parameters"""

  # Begin optimziation loop.
  all_tlosses = []
  all_elosses = []

  # Build some functions used in optimization.
  kl_warmup_fun = get_kl_warmup_fun(opt_hps)
  decay_fun = optimizers.exponential_decay(opt_hps['step_size'],
                                           opt_hps['decay_steps'],
                                           opt_hps['decay_factor'])

  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=opt_hps['adam_b1'],
                                                     b2=opt_hps['adam_b2'],
                                                     eps=opt_hps['adam_eps'])
  opt_state = opt_init(init_params)

  def update_w_gc(i, opt_state, hps, opt_hps, key, x_bxt,
                  kl_warmup):
    """Update fun for gradients, includes gradient clipping."""
    params = get_params(opt_state)
    grads = grad(lfads.training_loss_jit)(params, hps, key, x_bxt,
                                          kl_warmup, opt_hps['keep_rate'])
    clipped_grads = optimizers.clip_grads(grads, opt_hps['max_grad_norm'])
    return opt_update(i, clipped_grads, opt_state)

  update_w_gc_jit = jit(update_w_gc, static_argnums=(2,3))
 
  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = hps['batch_size']
  num_batches = opt_hps['num_batches']
  print_every = opt_hps['print_every']
  num_opt_loops = int(num_batches / print_every)
  params = get_params(opt_state)
  for oidx in range(num_opt_loops):
    batch_idx_start = oidx * print_every
    start_time = time.time()
    key, tkey, dtkey1, dtkey2, dekey1, dekey2 = \
        random.split(random.fold_in(key, oidx), 6)
    opt_state = optimize_core_jit(tkey, batch_idx_start,
                                  print_every, update_w_gc_jit, kl_warmup_fun,
                                  opt_state, hps, opt_hps,
                                  train_data_fun)
    batch_time = time.time() - start_time

    # Losses
    params = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    kl_warmup = kl_warmup_fun(batch_idx_start)
    # Training loss
    #didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    #x_bxt = train_data[didxs].astype(onp.float32)
    x_bxt = train_data_fun(dtkey1)
    tlosses = lfads.losses_jit(params, hps, dtkey2, x_bxt, kl_warmup, 1.0)

    # Evaluation loss
    #didxs = onp.random.randint(0, eval_data.shape[0], batch_size)
    #ex_bxt = eval_data[didxs].astype(onp.float32)
    ex_bxt = eval_data_fun(dekey1)
    elosses = lfads.losses_jit(params, hps, dekey2, ex_bxt, kl_warmup, 1.0)
    # Saving, printing.
    resps = softmax(params['prior']['resps'])
    rmin = onp.min(resps)
    rmax = onp.max(resps)
    rmean = onp.mean(resps)
    rstd = onp.std(resps)

    all_tlosses.append(tlosses)
    all_elosses.append(elosses)
    s1 = "Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}"
    s2 = "    Training losses {:0.0f} = NLL {:0.0f} + KL {:0.1f},{:0.1f} + L2 {:0.2f} + II L2 {:0.2f} + <II> {:0.2f} "
    s3 = "        Eval losses {:0.0f} = NLL {:0.0f} + KL {:0.1f},{:0.1f} + L2 {:0.2f} + II L2 {:0.2f} + <II> {:0.2f} "
    s4 = "        Resps: min {:0.4f}, mean {:0.4f}, max {:0.4f}, std {:0.4f}"
    print(s1.format(batch_idx_start+1, batch_pidx, batch_time,
                   decay_fun(batch_pidx)))
    print(s2.format(tlosses['total'], tlosses['nlog_p_xgz'],
                    tlosses['kl_prescale'], tlosses['kl'],
                    tlosses['l2'], tlosses['ii_l2'], tlosses['ii_mean']))
    print(s3.format(elosses['total'], elosses['nlog_p_xgz'],
                    elosses['kl_prescale'], elosses['kl'],
                    elosses['l2'], elosses['ii_l2'], tlosses['ii_mean']))
    print(s4.format(rmin, rmean, rmax, rstd))

    tlosses_thru_training = utils.merge_losses_dicts(all_tlosses)
    elosses_thru_training = utils.merge_losses_dicts(all_elosses)
    optimizer_details = {'tlosses' : tlosses_thru_training,
                         'elosses' : elosses_thru_training}

  return params, optimizer_details
