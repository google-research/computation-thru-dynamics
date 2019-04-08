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

import integrator_rnn_tutorial.utils as utils

MAX_SEED_INT = 10000000


def random_vrnn_params(key, u, n, o, g=1.0):
  """Generate random RNN parameters"""

  key, skeys = utils.keygen(key, 4)
  hscale = 0.1
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


def vrnn_run_with_h0(params, x_t, h0):
  """Run the Vanilla RNN T steps, where T is shape[0] of input."""
  # per-example predictions
  h = h0
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
batched_rnn_run_w_h0 = vmap(vrnn_run_with_h0, in_axes=(None, 0, 0))
  
  
def loss(params, inputs_bxtxu, targets_bxtxo, l2reg):
  """Compute the least squares loss of the output, plus L2 regularization."""
  _, outs_bxtxo = batched_rnn_run(params, inputs_bxtxu)
  flatten = lambda params: flatten_util.ravel_pytree(params)[0]
  l2_loss = l2reg * np.sum(flatten(params)**2)
  lms_loss = np.mean((outs_bxtxo - targets_bxtxo)**2)
  total_loss = lms_loss + l2_loss
  return {'total' : total_loss, 'lms' : lms_loss, 'l2' : l2_loss}


flatten = lambda params: flatten_util.ravel_pytree(params)[0]

def update_w_gc(i, opt_state, opt_update, x_bxt, f_bxt, max_grad_norm, l2reg):
  """Update the parameters w/ gradient clipped, gradient descent updates."""
  params = optimizers.get_params(opt_state)
  unflatten = flatten_util.ravel_pytree(params)[1] # Requires shape

  def training_loss(params, x_bxt, f_bxt, l2reg):
    return loss(params, x_bxt, f_bxt, l2reg)['total']
  
  grads = grad(training_loss)(params, x_bxt, f_bxt, l2reg)
  flat_grads = flatten(grads)
  grad_norm = np.sqrt(np.sum(flat_grads**2))
  normed_grads = np.where(grad_norm <= max_grad_norm, flat_grads,
                          flat_grads * (max_grad_norm / grad_norm))
  uf_grads = unflatten(normed_grads)
  return opt_update(i, uf_grads, opt_state)


loss_jit = jit(loss, static_argnums=(3,))
update_w_gc_jit = jit(update_w_gc, static_argnums=(2,5,6))


def run_trials(batched_run_fun, inputs_targets_h0s_fun, nbatches, batch_size):
  """Run a bunch of trials and save everything in a dictionary."""
  inputs = []
  hiddens = []
  outputs = []
  targets = []
  h0s = []
  for n in range(nbatches):
    data_seeds = onp.random.randint(0, MAX_SEED_INT, size=batch_size)
    keys = np.array([random.PRNGKey(ds) for ds in data_seeds])
    input_b, target_b, h0s_b = inputs_targets_h0s_fun(keys)
    if h0s_b is None:
      h_b, o_b = batched_run_fun(input_b)
    else:
      h_b, o_b = batched_run_fun(input_b, h0s_b)      
      h0s.append(h0s_b)
      
    inputs.append(input_b)
    hiddens.append(h_b)
    outputs.append(o_b)
    targets.append(target_b)

    
  trial_dict = {'inputs' : onp.vstack(inputs), 'hiddens' : onp.vstack(hiddens),
                'outputs' : onp.vstack(outputs), 'targets' : onp.vstack(targets)}
  if h0s_b is not None:
    trial_dict['h0s'] = onp.vstack(h0s)
  else:
    trial_dict['h0s'] = None
  return trial_dict

def plot_params(params):
  """ Plot the parameters of the vanilla RNN. """
  plt.figure(figsize=(16,8))
  plt.subplot(231)
  plt.stem(params['wO'][0,:])
  plt.title('wO - output weights')
  
  plt.subplot(232)
  plt.stem(params['h0'])
  plt.title('h0 - initial hidden state')
  
  plt.subplot(233)
  plt.imshow(params['wR'], interpolation=None)
  plt.title('wR - recurrent weights')
  plt.colorbar()
  
  plt.subplot(234)
  plt.stem(params['wI'])
  plt.title('wI - input weights')
  
  plt.subplot(235)
  plt.stem(params['bR'])
  plt.title('bR - recurrent biases')
  
  plt.subplot(236)
  evals, _ = onp.linalg.eig(params['wR'])
  x = onp.linspace(-1, 1, 1000)
  plt.plot(x, onp.sqrt(1-x**2), 'k')
  plt.plot(x, -onp.sqrt(1-x**2), 'k')
  plt.plot(onp.real(evals), onp.imag(evals), '.')
  plt.axis('equal')
  plt.title('Eigenvalues of wR')

  
def plot_examples(ntimesteps, rnn_internals, nexamples=1):
  """Plot some input/hidden/output triplets."""
  plt.figure(figsize=(nexamples*5, 12))
  for bidx in range(nexamples):
    plt.subplot(3, nexamples, bidx+1)
    plt.plot(rnn_internals['inputs'][bidx,:], 'k')
    plt.xlim([0, ntimesteps])
    plt.title('Example %d' % (bidx))
    if bidx == 0:
      plt.ylabel('Input')
      
  ntoplot = 10
  closeness = 0.25
  for bidx in range(nexamples):
    plt.subplot(3, nexamples, nexamples+bidx+1)
    plt.plot(rnn_internals['hiddens'][bidx, :, 0:ntoplot] +
             closeness * onp.arange(ntoplot), 'b')
    plt.xlim([0, ntimesteps])
    if bidx == 0:
      plt.ylabel('Hidden Units')
      
  for bidx in range(nexamples):
    plt.subplot(3, nexamples, 2*nexamples+bidx+1)
    plt.plot(rnn_internals['outputs'][bidx,:,:], 'r')
    plt.plot(rnn_internals['targets'][bidx,:,:], 'k')    
    plt.xlim([0, ntimesteps])
    plt.xlabel('Timesteps')
    if bidx == 0:
      plt.ylabel('Output')
