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


"""GRU functions for init, definition and running."""

from __future__ import print_function, division, absolute_import
import datetime
import h5py
from functools import partial

import jax.numpy as np
from jax import grad, jacrev, jit, lax, random, vmap
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy

import fixed_point_finder.utils as utils

MAX_SEED_INT = 10000000


def gru_params(key, **rnn_hps):
  """Generate GRU parameters

  Arguments:
    key: random.PRNGKey for random bits
    n: hidden state size
    u: input size
    i_factor: scaling factor for input weights
    h_factor: scaling factor for hidden -> hidden weights
    h_scale: scale on h0 initial condition

  Returns:
    a dictionary of parameters
  """
  key, skeys = utils.keygen(key, 6)
  u = rnn_hps['u']              # input
  n = rnn_hps['n']              # hidden
  o = rnn_hps['o']              # output
  
  ifactor = rnn_hps['i_factor'] / np.sqrt(u)
  hfactor = rnn_hps['h_factor'] / np.sqrt(n)
  hscale = rnn_hps['h_scale']
  
  wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
  wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)
  
  wCH = random.normal(next(skeys), (n,n)) * hfactor
  wCX = random.normal(next(skeys), (n,u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  # Include the readout params in the GRU, though technically
  # not a part of the GRU.
  pfactor = 1.0 / np.sqrt(n)
  wO = random.normal(next(skeys), (o,n)) * pfactor
  bO = np.zeros((o,))
  return {'h0' : random.normal(next(skeys), (n,)) * hscale,
          'wRUHX' : wRUHX,
          'wCHX' : wCHX,
          'bRU' : np.zeros((n+n,)),
          'bC' : np.zeros((n,)),
          'wO' : wO,
          'bO' : bO}


def sigmoid(x):
  """ Implement   1 / ( 1 + exp( -x ) )   in terms of tanh."""
  return 0.5 * (np.tanh(x / 2.) + 1)


def gru(params, h, x, bfg=0.5):
  """Implement the GRU equations.

  Arguments:
    params: dictionary of GRU parameters
    h: np array of  hidden state
    x: np array of input
    bfg: bias on forget gate (useful for learning if > 0.0)

  Returns:
    np array of hidden state after GRU update"""

  hx = np.concatenate([h, x], axis=0)
  ru = sigmoid(np.dot(params['wRUHX'], hx) + params['bRU'])
  r, u = np.split(ru, 2, axis=0)
  rhx = np.concatenate([r * h, x])
  c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'] + bfg)
  return u * h + (1.0 - u) * c


def gru_scan(params, h, x, bfg=0.5):
  """Return the output twice for scan."""
  h = gru(params, h, x, bfg)
  return h, h


def affine(params, x):
  """Implement y = w x + b

  Args: 
    params: dictionary of affine parameters
    x: np array of input"""
  return np.dot(params['wO'], x) + params['bO']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims) So
# map over first dimension to hand t_x_m.  I.e. if affine yields n_y_1
# = dot(n_W_m, m_x_1), then batch_affine yields t_y_n.
batch_affine = vmap(affine, in_axes=(None, 0))


def gru_run_with_h0(params, x_t, h0):
  """Run the Vanilla RNN T steps, where T is shape[0] of input.

  Args:
    params: dict of GRU parameters
    x_t: np array of inputs with dim ntime x u
    h0: initial condition for hidden state

  Returns: 
    2-tuple of np arrays (hidden states w dim ntime x n, outputs w dim ntim x o)
  """

  f = partial(gru_scan, params)
  _, h_t = lax.scan(f, h0, x_t)
  o_t = batch_affine(params, h_t)
  return h_t, o_t  


def gru_run(params, x_t):
  """Run the Vanilla RNN T steps, where T is shape[0] of input.
  
  Args:
    params: dict of GRU parameters
    x_t: np array of inputs with dim ntime x u

  Returns: 
    2-tuple of np arrays (hidden states w dim ntime x n, outputs w dim ntim x o)
  """  
  return gru_run_with_h0(params, x_t, params['h0'])

  
# Let's make it handle batches using `vmap`
batched_rnn_run = vmap(gru_run, in_axes=(None, 0))
batched_rnn_run_w_h0 = vmap(gru_run_with_h0, in_axes=(None, 0, 0))
  
  
def loss(params, inputs_bxtxu, targets_bxtxo, targets_mask_t, l2reg):
  """Compute the least squares loss of the output, plus L2 regularization.

  Args: 
    params: dict RNN parameters
    inputs_bxtxu: np array of inputs batch x time x input dim
    targets_bxtxo: np array of targets batx x time x output dim
    targets_mask_t: list of time indices where target is active
    l2reg: float, hyper parameter controlling strength of L2 regularization
  
  Returns:
    dict of losses
  """
  _, outs_bxtxo = batched_rnn_run(params, inputs_bxtxu)
  l2_loss = l2reg * optimizers.l2_norm(params)**2
  outs_bxsxo = outs_bxtxo[:, targets_mask_t, :]
  targets_bxsxo = targets_bxtxo[:, targets_mask_t, :]
  lms_loss = np.mean((outs_bxsxo - targets_bxsxo)**2)
  total_loss = lms_loss + l2_loss
  return {'total' : total_loss, 'lms' : lms_loss, 'l2' : l2_loss}


def update_w_gc(i, opt_state, opt_update, get_params,
                x_bxt, f_bxt, f_mask_bxt, max_grad_norm, l2reg):
  """Update the parameters w/ gradient clipped, gradient descent updates.

  Arguments: 
    i: batch number
    opt_state: parameters plus optimizer state
    opt_update: optimizer state update function
    get_params: function to extract parameters from optimizer state
    x_bxt: rnn inputs
    f_bxt: rnn targets
    f_mask_bxt: masks for when target is defined
    max_grad_norm: maximum norm value gradient is allowed to take
    l2reg: l2 regularization hyperparameter
  
  Returns: 
    opt_state tuple (as above) that includes updated parameters and optimzier 
      state.
  """
  params = get_params(opt_state)

  def training_loss(params, x_bxt, f_bxt, l2reg):
    return loss(params, x_bxt, f_bxt, f_mask_bxt, l2reg)['total']
  
  grads = grad(training_loss)(params, x_bxt, f_bxt, l2reg)
  clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
  return opt_update(i, clipped_grads, opt_state)


loss_jit = jit(loss)
update_w_gc_jit = jit(update_w_gc, static_argnums=(2,3))


def run_trials(batched_run_fun, io_fun, nbatches, batch_size):
  """Run a bunch of trials and save everything in a dictionary.
  
  Args: 
    batched_run_fun: function for running rnn in batch with signature either
      inputs -> hiddens, outputs OR
      inputs, h0s -> hiddens, outputs
    io_fun: function for creating inputs, targets, masks 
      and initial conditions. initial conditions may be None and the parameter 
      is used.  
    nbatches: Number of batches to run
    batch_size: Size of batch to run

  Returns: 
    A dictionary with the trial structure, keys are: 
      inputs, hiddens, outputs and targets, each an np array with dim
      nbatches*batch_size * ntimesteps * dim 
  """
  inputs = []
  hiddens = []
  outputs = []
  targets = []
  h0s = []
  for n in range(nbatches):
    data_seeds = onp.random.randint(0, MAX_SEED_INT, size=batch_size)
    keys = np.array([random.PRNGKey(ds) for ds in data_seeds])
    input_b, target_b, masks_b, h0s_b = io_fun(keys)
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
  """Plot the parameters of the GRU.

  Args: 
    params: Parmeters of the GRU
  """
  plt.figure(figsize=(16,8))
  plt.subplot(231)
  plt.stem(params['wO'][0,:])
  plt.title('wO - output weights')
  
  plt.subplot(232)
  plt.stem(params['h0'])
  plt.title('h0 - initial hidden state')
  
  plt.subplot(233)
  plt.imshow(params['wRUHX'], interpolation=None)
  plt.title('wRUHX - recurrent weights')
  plt.colorbar()
  
  plt.subplot(234)
  plt.imshow(params['wCHX'], interpolation=None)
  plt.title('wCHX')
  
  plt.subplot(235)
  plt.stem(params['bRU'])
  plt.title('bRU - recurrent biases')
  
  plt.subplot(236)
  xdim = 1
  rnn_fun_h = lambda h : gru(params, h, np.zeros(xdim))
  dFdh = jacrev(rnn_fun_h)(params['h0'])
  evals, _ = onp.linalg.eig(dFdh)
  x = onp.linspace(-1, 1, 1000)
  plt.plot(x, onp.sqrt(1-x**2), 'k')
  plt.plot(x, -onp.sqrt(1-x**2), 'k')
  plt.plot(onp.real(evals), onp.imag(evals), '.')
  plt.axis('equal')
  plt.xlabel('Real($\lambda$)')
  plt.ylabel('Imaginary($\lambda$)')
  plt.title('Eigenvalues of $dF/dh(h_0)$')

  
def plot_examples(ntimesteps, rnn_internals, nexamples=1):
  """Plot some input/hidden/output triplets.
  
  Args: 
    ntimesteps: Number of time steps to plot
    rnn_internals: dict returned by run_trials.
  """
  plt.figure(figsize=(nexamples*5, 16))
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
