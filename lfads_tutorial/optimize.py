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
from jax import grad, jit, random, vmap
from jax import jacrev, jacfwd
from jax.experimental import optimizers
import jax.flatten_util as flatten_util

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

import lfads
import utils

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
  kl_max = lfads_opt_hps['kl_max']
  def kl_warmup(batch):
    if batch < kl_warmup_start:
      return 0.0
    elif batch > kl_warmup_end:
      return kl_max
    else:
      return kl_max * (onp.float(batch - kl_warmup_start) /
                      onp.float(kl_warmup_end - kl_warmup_start))
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


def optimize_lfads(init_params, lfads_hps, lfads_opt_hps,
                   train_data, eval_data):
  """Optimize the LFADS model and print batch based optimization data.

  Arguments:
    init_params: a dict of parameters to be trained
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training
    eval_data: nexamples x time x ndims np array of data for evaluation

  Returns:
    a dictionary of trained parameters"""

  batch_size = lfads_hps['batch_size']
  num_batches = lfads_opt_hps['num_batches']
  print_every = lfads_opt_hps['print_every']

  # Build some functions used in optimization.
  kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
  decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],
                                           lfads_opt_hps['decay_steps'],
                                           lfads_opt_hps['decay_factor'])
  opt_init, opt_update = optimizers.adam(step_size=decay_fun,
                                         b1=lfads_opt_hps['adam_b1'],
                                         b2=lfads_opt_hps['adam_b2'],
                                         eps=lfads_opt_hps['adam_eps'])
  update_w_gc = get_update_w_gc_fun(init_params, opt_update)
  update_w_gc_jit = jit(update_w_gc, static_argnums=(2, 3))

  # Begin optimziation loop.
  start_time = time.time()
  opt_state = opt_init(init_params)
  for bidx in range(num_batches):
    kl_warmup = kl_warmup_fun(bidx)
    didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    x_bxt = train_data[didxs].astype(onp.float32)
    key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
    opt_state = update_w_gc_jit(bidx, opt_state, lfads_hps, lfads_opt_hps,
                                key, x_bxt, kl_warmup)

    if bidx % print_every == 0:
      params = optimizers.get_params(opt_state)

      # Training loss
      didxs = onp.random.randint(0, train_data.shape[0], batch_size)
      x_bxt = train_data[didxs].astype(onp.float32)
      key = random.P2RNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
      tlosses = lfads_losses_jit(params, lfads_hps, key, x_bxt, kl_warmup, 1.0)

      # Evaluation loss
      key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
      didxs = onp.random.randint(0, eval_data.shape[0], batch_size)
      ex_bxt = eval_data[didxs].astype(onp.float32)
      # Commented out lfads_eval_losses_jit cuz freezing.
      elosses = lfads_losses_jit(params, lfads_hps_eval, key, ex_bxt,
                                 kl_warmup, 1.0)
      # Saving, printing.
      all_tlosses.append(tlosses)
      all_elosses.append(elosses)
      batch_time = time.time() - start_time
      s = "Batch {} in {:0.2f} sec, Step size: {:0.5f}, \
              Training loss {:0.0f}, Eval loss {:0.0f}"
      print(s.format(bidx, batch_time, decay_fun(batch),
                     tlosses['total'], elosses['total']))
      start_time = time.time()

  return optimizers.get_params(opt_state), tlosses, elosses
