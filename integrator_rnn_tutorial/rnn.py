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


"""Vanilla RNN functions for init, definition and running."""

from __future__ import print_function, division, absolute_import
import datetime
import h5py

import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacrev, jacfwd
from jax.experimental import optimizers
import jax.experimental.stax as stax
import jax.flatten_util as flatten_util

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import os
import time

import utils

MAX_SEED_INT = 10000000


def random_vrnn_params(key, u, n, o, g=1.0):
  """Generate random RNN parameters"""

  key, skeys = utils.keygen(key, 4)
  hscale = 0.25
  ifactor = 1.0 / np.sqrt(u)
  hfactor = g / np.sqrt(n)
  pfactor = 1.0 / np.sqrt(n)
  return {'h0' : random.normal(next(skeys), (n,)) * hscale,
          'wI' : random.normal(next(skeys), (n,u)) * ifactor,
          'wR' : random.normal(next(skeys), (n,n)) *  hfactor,
          'wO' : random.normal(next(skeys), (o,n)) * pfactor,
          'bR' : np.zeros([n]),
          'bO' : np.zeros([o])}


def affine(params, x):
  """Implement y = w x + b"""
  return np.dot(params['wO'], x) + params['bO']

# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims)
# So map over first dimension to hand t_x_m.
# I.e. if affine yields n_y_1 = dot(n_W_m, m_x_1), then
# batch_affine yields t_y_n.
# And so the vectorization pattern goes for all batch_* functions.
batch_affine = vmap(affine, in_axes=(None, 0))


def vrnn(params, h, x):
  """Run the Vanilla RNN one step"""
  a = np.dot(params['wI'], x) + params['bR'] + np.dot(params['wR'], h)
  return np.tanh(a)


def vrnn_run(params, x_t):
  """Run the Vanilla RNN T steps, where T is shape[0] of input."""
  # per-example predictions
  h = params['h0']
  h_t = []
  for x in x_t:
    h = vrnn(params, h, x)
    h_t.append(h)
    
  h_t = np.array(h_t)  
  o_t = batch_affine(params, h_t)
  return h_t, o_t

  
# Let's upgrade it to handle batches using `vmap`
# Make a batched version of the `predict` function
batched_rnn_run = vmap(vrnn_run, in_axes=(None, 0))
  
  
def loss(params, inputs_bxtxu, targets_bxtxo, l2reg):
  """Compute the least squares loss of the output, plus L2 regularization."""
  _, outs_bxtxo = batched_rnn_run(params, inputs_bxtxu)
  flatten = lambda params: flatten_util.ravel_pytree(params)[0]
  l2_loss = l2reg * np.sum(flatten(params)**2)
  lms_loss = np.mean((outs_bxtxo - targets_bxtxo)**2)
  return lms_loss + l2_loss


flatten = lambda params: flatten_util.ravel_pytree(params)[0]

def update_w_gc(i, opt_state, opt_update, x_bxt, f_bxt, max_grad_norm, l2reg):
  """Update the parameters w/ gradient clipped, gradient descent updates."""
  params = optimizers.get_params(opt_state)
  unflatten = flatten_util.ravel_pytree(params)[1] # Requires shape

  grads = grad(loss)(params, x_bxt, f_bxt, l2reg)
  flat_grads = flatten(grads)
  grad_norm = np.sqrt(np.sum(flat_grads**2))
  normed_grads = np.where(grad_norm <= max_grad_norm, flat_grads,
                          flat_grads * (max_grad_norm / grad_norm))
  uf_grads = unflatten(normed_grads)
  return opt_update(i, uf_grads, opt_state)


loss_jit = jit(loss, static_argnums=(3,))
update_w_gc_jit = jit(update_w_gc, static_argnums=(2,5,6))


def run_trials(batched_run_fun, inputs_and_targets_fun, nbatches, batch_size):
  """Run a bunch of trials and save everything in a dictionary."""
  inputs = []
  hiddens = []
  outputs = []
  targets = []
  for n in range(nbatches):
    data_seeds = onp.random.randint(0, MAX_SEED_INT, size=batch_size)
    keys = np.array([random.PRNGKey(ds) for ds in data_seeds])
    input, target = inputs_and_targets_fun(keys)
    h_bxt, o_bxt = batched_run_fun(input)

    inputs.append(input)
    hiddens.append(h_bxt)
    outputs.append(o_bxt)
    targets.append(target)

  return {'inputs' : onp.vstack(inputs), 'hiddens' : onp.vstack(hiddens),
          'outputs' : onp.vstack(outputs), 'targets' : onp.vstack(targets)}

