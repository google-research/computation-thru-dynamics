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


"""LFADS architecture and loss functions."""

# TODO(sussillo): tanh(g0) and tanh(ii) optional

from __future__ import print_function, division, absolute_import
from functools import partial

import jax.numpy as np
from jax import jit, lax, random, vmap
from jax.experimental import optimizers

#import lfads_tutorial.distributions as dists
#import lfads_tutorial.utils as utils


def sigmoid(x):
  return 0.5 * (np.tanh(x / 2.) + 1)


def gmm_params(key, size, dim, mean_std=0.3, var_mean=0.2, var_std=0.025):
  """Params for Gaussian mixture model.

  Arguments:
    key: random.PRNGKey for random bits
    num: number of gaussians in mixture
    dim: dimension of gaussian
    mean_scale: standard deviation of random means around 0
    var_mean : mean of variance of each gaussian
    var_std: standard deviation of the mean of each gaussian

  Returns:
    a dictionary of mixture model parameters
  """
  keys = random.split(key, 2)
  resp_logits = np.ones((num,))       # always put into softmax
  mean_params = mean_scale * random.normal(keys[0], shape=(num, dim))
  logvar_params = np.log(var_mean * np.ones((num, dim))
                         + var_std * random.normal(keys[1], shape=(num, dim)))
  return {'resp' : resp_logits, 'mean' : mean_params, 'logvar' : logvar_params}


def linear_params(key, o, u, ifactor=1.0):
  """Params for y = w x

  Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  keys = random.split(key, 2)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(keys[0], (o, u)) * ifactor}


def affine_params(key, o, u, ifactor=1.0):
  """Params for y = w x + b

  Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  keys = random.split(key, 2)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(keys[0], (o, u)) * ifactor,
          'b' : np.zeros((o,))}


def gru_params(key, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
  """Generate GRU parameters

  Arguments:
    key: random.PRNGKey for random bits
    n: hidden state size
    u: input size
    ifactor: scaling factor for input weights
    hfactor: scaling factor for hidden -> hidden weights
    hscale: scale on h0 initial condition

  Returns:
    a dictionary of parameters
  """
  keys = random.split(key, 5)
  ifactor = ifactor / np.sqrt(u)
  hfactor = hfactor / np.sqrt(n)

  wRUH = random.normal(keys[0], (n+n,n)) * hfactor
  wRUX = random.normal(keys[1], (n+n,u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)

  wCH = random.normal(keys[2], (n,n)) * hfactor
  wCX = random.normal(keys[3], (n,u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  return {'h0' : random.normal(keys[4], (n,)) * hscale,
          'wRUHX' : wRUHX,
          'wCHX' : wCHX,
          'bRU' : np.zeros((n+n,)),
          'bC' : np.zeros((n,))}


def affine(params, x):
  """Implement y = w x + b

  Arguments:
    params: a dictionary of params
    x: np array of input

  Returns:
    np array of output
  """
  return np.dot(params['w'], x) + params['b']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims)
# So map over first dimension to hand t_x_m.
# I.e. if affine yields n_y_1 = dot(n_W_m, m_x_1), then
# batch_affine yields t_y_n.
# And so the vectorization pattern goes for all batch_* functions.
batch_affine = vmap(affine, in_axes=(None, 0))


def normed_linear(params, x):
  """Implement y = \hat{w} x, where \hat{w}_ij = w_ij / |w_{i:}|, norm over j

  Arguments:
    params: a dictionary of params
    x: np array of input

  Returns:
    np array of output
  """
  w = params['w']
  w_row_norms = np.sqrt(np.sum(w**2, axis=1, keepdims=True))
  w = w / w_row_norms
  return np.dot(w, x)


# Note not BatchNorm, the neural network regularizer,
# rather just batching the normed linear function above.
batch_normed_linear = vmap(normed_linear, in_axes=(None, 0))


def dropout(x, key, keep_rate):
  """Implement a dropout layer.

  Arguments:
    x: np array to be dropped out
    key: random.PRNGKey for random bits
    keep_rate: dropout rate

  Returns:
    np array of dropped out x
  """
  # The shenanigans with np.where are to avoid having to re-jit if
  # keep rate changes.
  keys = random.split(key, 2)
  do_keep = random.bernoulli(keys[0], keep_rate, x.shape)
  kept_rates = np.where(do_keep, x / keep_rate, 0.0)
  return np.where(keep_rate < 1.0, kept_rates, x)


# Note that dropout is a feed-forward routine that requires randomness. Thus,
# the keys argument is also vectorized over, and you'll see the correct
# number of keys allocated by the caller.
batch_dropout = vmap(dropout, in_axes=(0, 0, None))


def run_dropout(x_t, key, keep_rate):
  """Run the dropout layer over additional dimensions, e.g. time.

  Arguments:
    x_t: np array to be dropped out
    key: random.PRNGKey for random bits
    keep_rate: dropout rate

  Returns:
    np array of dropped out x
  """
  ntime = x_t.shape[0]
  keys = random.split(key, ntime)
  return batch_dropout(x_t, keys, keep_rate)


def gru(params, h, x):
  """Implement the GRU equations.

  Arguments:
    params: dictionary of GRU parameters
    h: np array of  hidden state
    x: np array of input

  Returns:
    np array of hidden state after GRU update"""
  bfg = 0.5
  hx = np.concatenate([h, x], axis=0)
  ru = sigmoid(np.dot(params['wRUHX'], hx) + params['bRU'])
  r, u = np.split(ru, 2, axis=0)
  rhx = np.concatenate([r * h, x])
  c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'] + bfg)
  return u * h + (1.0 - u) * c


def make_rnn_for_scan(rnn, params):
  """Scan requires f(h, x) -> h, h, in this application.
  Args:
    rnn : f with sig (params, h, x) -> h
    params: params in f() sig.

  Returns:
    f adapted for scan
  """
  def rnn_for_scan(h, x):
    h = rnn(params, h, x)
    return h, h
  return rnn_for_scan


def run_rnn(rnn_for_scan, x_t, h0):
  """Run an RNN module forward in time.

  Arguments:
    rnn_for_scan: function for running RNN one step (h, x) -> (h, h)
      The params already embedded in the function.
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn

  Returns:
    np array of rnn applied to time data with leading dim being time"""
  _, h_t = lax.scan(rnn_for_scan, h0, x_t)
  return h_t


def run_bidirectional_rnn(params, fwd_rnn, bwd_rnn, x_t):
  """Run an RNN encoder backwards and forwards over some time series data.

  Arguments:
    params: a dictionary of bidirectional RNN encoder parameters
    fwd_rnn: function for running forward rnn encoding
    bwd_rnn: function for running backward rnn encoding
    x_t: np array data for RNN input with leading dim being time

  Returns:
    tuple of np array concatenated forward, backward encoding, and
      np array of concatenation of [forward_enc(T), backward_enc(1)]
  """
  fwd_rnn_scan = make_rnn_for_scan(fwd_rnn, params['fwd_rnn'])
  bwd_rnn_scan = make_rnn_for_scan(bwd_rnn, params['bwd_rnn'])

  fwd_enc_t = run_rnn(fwd_rnn_scan, x_t, params['fwd_rnn']['h0'])
  bwd_enc_t = np.flipud(run_rnn(bwd_rnn_scan, np.flipud(x_t),
                                params['bwd_rnn']['h0']))
  full_enc = np.concatenate([fwd_enc_t, bwd_enc_t], axis=1)
  enc_ends = np.squeeze(np.concatenate([bwd_enc_t[0:1], fwd_enc_t[-1:]], axis=1), axis=0)
  return full_enc, enc_ends


def lfads_params(key, lfads_hps):
  """Instantiate random LFADS parameters.

  Arguments:
    key: random.PRNGKey for random bits
    lfads_hps: a dict of LFADS hyperparameters

  Returns:
    a dictionary of LFADS parameters
  """
  keys = random.split(key, 8)

  data_dim = lfads_hps['data_dim']
  ntimesteps = lfads_hps['ntimesteps']
  enc_dim = lfads_hps['enc_dim']
  con_dim = lfads_hps['con_dim']
  ii_dim = lfads_hps['ii_dim']
  gen_dim = lfads_hps['gen_dim']
  factors_dim = lfads_hps['factors_dim']
  batch_size = lfads_hps['batch_size']
  ic_dim = enc_dim              # Could make a different HP via a linear layer
  z_dim = ic_dim + ntimesteps * ii_dim
  gmm_size = lfads_hps['gmm_size']

  ic_enc_params = {'fwd_rnn' : gru_params(keys[0], enc_dim, data_dim),
                   'bwd_rnn' : gru_params(keys[1], enc_dim, data_dim)}
  gen_ic_params = affine_params(keys[2], 2*gen_dim, 2*enc_dim) # m, v <- bi
  prior_params = gmm_params(keys[3], gmm_size, z_dim)
  con_params = gru_params(keys[4], con_dim, 2*enc_dim + factors_dim + ii_dim)
  con_out_params = affine_params(keys[5], 2*ii_dim, con_dim)   #m, v
  gen_params = gru_params(keys[6], gen_dim, ii_dim)
  factors_params = linear_params(keys[7], factors_dim, gen_dim)
  lograte_params = affine_params(keys[8], data_dim, factors_dim)

  return {'ic_enc' : ic_enc_params,
          'gen_ic' : gen_ic_params,
          'con' : con_params, 'con_out' : con_out_params,
          'prior' : prior_params,
          'gen' : gen_params, 'factors' : factors_params,
          'f0' : np.zeros((lfads_hps['factors_dim'],)),
          'ii0' : np.zeros((lfads_hps['ii_dim'],)),
          'logrates' : lograte_params}


def lfads_encode(params, lfads_hps, key, x_t, keep_rate):
  """Run the LFADS network from input to generator initial condition vars.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_t: np array input for lfads with leading dimension being time
    keep_rate: dropout keep rate

  Returns:
    3-tuple of np arrays: generator initial condition mean, log variance
      and also bidirectional encoding of x_t, with leading dim being time
  """
  keys = random.split(key, 3)

  # Encode the input
  x_t = run_dropout(x_t, keys[0], keep_rate)
  con_ins_t, gen_pre_ics = run_bidirectional_rnn(params['ic_enc'],
                                                 gru, gru, x_t)
  # Push through to posterior mean and variance for initial conditions.
  xenc_t = dropout(con_ins_t, keys[1], keep_rate)
  gen_pre_ics = dropout(gen_pre_ics, keys[2], keep_rate)
  ic_gauss_params = affine(params['gen_ic'], gen_pre_ics)
  ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
  return ic_mean, ic_logvar, xenc_t


def lfads_decode_one_step(params, lfads_hps, key, keep_rate, c, ii, f, g, xenc):
  """Run the LFADS network from latent variables to log rates one time step.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    keep_rate: dropout keep rate
    c: controller state at time step t-1
    g: generator state at time step t-1
    ii: inferred input from time step t-1 (autoregressive sampling here)
    f: factors at time step t-1
    xenc: np array bidirectional encoding at time t of input (x_t)

  Returns:
    7-tuple of np arrays all with leading dim being time,
      controller hidden state, generator hidden state, factors,
      inferred input (ii) sample, ii mean, ii log var, log rates
  """
  keys = random.split(key, 2)
  cin = np.concatenate([xenc, f, ii], axis=0)
  c = gru(params['con'], c, cin)
  cout = affine(params['con_out'], c)
  ii_mean, ii_logvar = np.split(cout, 2, axis=0) # inferred input params
  ii = np.tanh(diag_gaussian_sample(keys[0], ii_mean, ii_logvar, lfads_hps['var_min']))
  g = gru(params['gen'], g, ii)
  g = dropout(g, keys[1], keep_rate)
  f = normed_linear(params['factors'], g)
  lograte = affine(params['logrates'], f)
  return c, g, f, ii, ii_mean, ii_logvar, lograte


def lfads_decode_prior_one_step(params, lfads_hps, key, ii, f, g):
  """Run the LFADS network from latent variables to log rates one time step.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    g: generator state at time step t-1
    ii: inferred input from time step t-1 (autoregressive sampling here)
    f: factors at time step t-1

  Returns:
    7-tuple of np arrays all with leading dim being time,
      controller hidden state, generator hidden state, factors,
      inferred input (ii) sample, ii mean, ii log var, log rates
  """
  g = gru(params['gen'], g, ii) # ii tanh'd at sample of multidim gaussian
  f = normed_linear(params['factors'], g)
  lograte = affine(params['logrates'], f)
  return g, f, ii, lograte


def lfads_decode_one_step_scan(params, lfads_hps, keep_rate, state, key_n_xenc):
  """Run the LFADS network one step, prepare the inputs and outputs for scan.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    keep_rate: dropout keep rate
    state: (controller state at time step t-1, generator state at time step t-1,
            factors at time step t-1)
    key_n_xenc: (random key, np array bidirectional encoding at time t of input (x_t))

  Returns: 2-tuple of state and state plus returned values
    ((controller state, inferred inputs, generator state, factors),
    (7-tuple of np arrays all with leading dim being time,
      controller hidden state, generator hidden state, factors,
      inferred input (ii) sample, ii mean, ii log var,
      log rate))
  """
  key, xenc = key_n_xenc
  c, ii, g, f = state
  state_and_returns = lfads_decode_one_step(params, lfads_hps, key, keep_rate,
                                            c, ii, f, g, xenc)
  c, g, f, ii, ii_mean, ii_logvar, lograte = state_and_returns
  state = (c, ii, g, f)
  return state, state_and_returns


def lfads_decode_prior_one_step_scan(params, lfads_hps, state, key_n_ii):
  """Run the LFADS network one step, prepare the inputs and outputs for scan.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    state: (controller state at time step t-1, generator state at time step t-1,
            factors at time step t-1)
    key_n_ii: (random key, inferred input sample)

  Returns: 2-tuple of state and state plus returned values
    ((inferred inputs, generator state, factors),
    (4-tuple of np arrays all with leading dim being time,
      generator hidden state, factors, inferred input (ii) sample, log rate))
  """
  key, ii = key_n_ii
  _, g, f = state
  state_and_returns = lfads_decode_prior_one_step(params, lfads_hps, key,
                                                  ii, f, g)
  g, f, ii, lograte = state_and_returns
  state = (ii, g, f)
  return state, state_and_returns


def lfads_decode(params, lfads_hps, key, ic_mean, ic_logvar, xenc_t, keep_rate):
  """Run the LFADS network from latent variables to log rates.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    ic_mean: np array of generator initial condition mean
    ic_logvar: np array of generator initial condition log variance
    xenc_t: np array bidirectional encoding of input (x_t) with leading dim
      being time
    keep_rate: dropout keep rate

  Returns:
    7-tuple of np arrays all with leading dim being time,
      controller hidden state, inferred input mean, inferred input log var,
      generator hidden state, factors and log rates
  """

  keys = random.split(key, 2)

  # Since the factors feed back to the controller,
  #    factors_{t-1} -> controller_t -> sample_t -> generator_t -> factors_t
  # is really one big loop and therefor one RNN.
  ii0 = params['ii0']
  c0 = params['con']['h0']
  g0 = np.tanh(diag_gaussian_sample(keys[0], ic_mean, ic_logvar,
                                    lfads_hps['var_min']))
  f0 = params['f0'] # np.zeros((lfads_hps['factors_dim'],))

  # Make all the randomness for all T steps at once, it's more efficient.
  # The random keys get passed into scan along with the input, so the input
  # becomes of a 2-tuple (keys, actual input).
  T = xenc_t.shape[0]
  keys_t = random.split(keys[1], T)

  state0 = (c0, ii0, g0, f0)
  decoder = partial(lfads_decode_one_step_scan, *(params, lfads_hps, keep_rate))
  _, state_and_returns_t = lax.scan(decoder, state0, (keys_t, xenc_t))
  return state_and_returns_t



def compose_sample(lfads_hps, ic_j, ii_txi):
  """Compose latent code from initial condition and inferred input."""
  ii_ti = np.reshape(ii_txi, (-1,))
  return np.concatenate([ic_j, ii_ti], axis=0)


batch_compose_sample = vmap(compose_sample, in_axes=(None, 0, 0))


def decompose_sample(lfads_hps, z):
  """Break apart the latent code into initial condition and inferred input."""
  ic_dim = lfads_hps['enc_dim']
  ic_j, ii_ti = np.split(z, (ic_dim,), axis=0)
  ii_txi = np.reshape(ii_ti, (-1, lfads_hps['ii_dim']))
  return ic_j, ii_txi


def batch_decompose_sample(lfads_hps, z_cxz):
  ic_dim = lfads_hps['enc_dim']
  ic_cxj, ii_cxti = np.split(z_cxz, (ic_dim,), axis=1)
  ii_cxtxi = np.reshape(ii_cxti, (ii_cxti.shape[0], -1, lfads_hps['ii_dim']))
  return ic_cxj, ii_cxtxi


def lfads_decode_prior(params, lfads_hps, key, z_sample):
  """Run the LFADS network from latent variables to log rates.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    z_sample: sample of the latents a numpy array composed of initial condition
      and inferred inputs
  Returns:
    4-tuple of np arrays all with leading dim being time,
      (generator hidden state, factors, inferred input, and log rates)
  """

  g0, ii_txi = decompose_sample(lfads_hps, z_sample)
  g0 = np.tanh(g0)
  ii_txi = np.tanh(ii_txi)
  # Since the factors feed back to the controller,
  #    factors_{t-1} -> controller_t -> sample_t -> generator_t -> factors_t
  # is really one big loop and therefor one RNN.
  ii0 = params['ii0']
  f0 = params['f0'] # np.zeros((lfads_hps['factors_dim'],))

  # Make all the randomness for all T steps at once, it's more efficient.
  # The random keys get passed into scan along with the input, so the input
  # becomes of a 2-tuple (keys, actual input).
  T = ii_txi.shape[0]
  keys = random.split(key, 2)
  keys_t = random.split(keys[0], T)

  state0 = (ii0, g0, f0)
  decoder = partial(lfads_decode_prior_one_step_scan, *(params, lfads_hps))
  _, state_and_returns_t = lax.scan(decoder, state0, (keys_t, ii_txi))
  return state_and_returns_t


def lfads(params, lfads_hps, key, x_t, keep_rate):
  """Run the LFADS network from input to output.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_t: np array of input with leading dim being time
    keep_rate: dropout keep rate

  Returns:
    A dictionary of np arrays of all LFADS values of interest.
  """

  keys = random.split(key, num=2)

  ic_mean, ic_logvar, xenc_t = \
      lfads_encode(params, lfads_hps, keys[0], x_t, keep_rate)

  c_t, gen_t, factor_t, ii_t, ii_mean_t, ii_logvar_t, lograte_t = \
      lfads_decode(params, lfads_hps, keys[1], ic_mean, ic_logvar,
                   xenc_t, keep_rate)

  # As this is tutorial code, we're passing everything around.
  return {'xenc_t' : xenc_t, 'ic_mean' : ic_mean, 'ic_logvar' : ic_logvar,
          'ii_t' : ii_t, 'c_t' : c_t, 'ii_mean_t' : ii_mean_t,
          'ii_logvar_t' : ii_logvar_t, 'gen_t' : gen_t, 'factor_t' : factor_t,
          'lograte_t' : lograte_t}


def lfads_prior_sample(params, lfads_hps, key):
  """Run the LFADS network from input to output.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits

  Returns:
    A dictionary of np arrays of all LFADS values of interest. Note this is a
    reduced set compared to running lfads with an encoder because the prior
    samples from latents and then runs only the generative model.
  """
  keys = random.split(key, num=2)
  z_sample = gmm_sample(keys[0], params['gmm_resp'], params['gmm_mean'],
                        params['gmm_logvar'], 1e-16)

  gen_t, factor_t, ii_t, lograte_t = \
      lfads_decode_prior(params, lfads_hps, keys[1], z_sample)

  return {'ii_t' : ii_t, 'gen_t' : gen_t, 'factor_t' : factor_t,
          'lograte_t' : lograte_t}


lfads_encode_jit = jit(lfads_encode)
lfads_decode_jit = jit(lfads_decode, static_argnums=(1,))
lfads_jit = jit(lfads, static_argnums=(1,))


# Batching accomplished by vectorized mapping. We simultaneously map over random
# keys for forward-pass randomness and inputs for batching.
batch_lfads = vmap(lfads, in_axes=(None, None, 0, 0, None))
batch_lfads_prior_sample = vmap(lfads_prior_sample, in_axes=(None, None, 0))


def lfads_losses(params, lfads_hps, key, x_bxt, kl_scale, keep_rate):
  """Compute the training loss of the LFADS autoencoder

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    keep_rate: dropout keep rate
    kl_scale: scale on KL

  Returns:
    a dictionary of all losses, including the key 'total' used for optimization
  """

  B = lfads_hps['batch_size']
  C = lfads_hps['gmm_size']
  S = lfads_hps['gmm_num_samples']
  I = lfads_hps['ii_dim']
  T = lfads_hps['ntimesteps']
  ic_dim = lfads_hps['enc_dim'] # didn't implement linear transformation yet

  keys = random.split(key, 3)
  keys_b = random.split(keys[0], B)
  lfads = batch_lfads(params, lfads_hps, keys_b, x_bxt, keep_rate)

  post_mean_bxz = batch_compose_sample(lfads_hps, lfads['ic_mean'],
                                       lfads['ii_mean_t'])
  post_logvar_bxz = batch_compose_sample(lfads_hps, lfads['ic_logvar'],
                                         lfads['ii_logvar_t'])

  # Sum over time and state dims, average over batch.
  # KL - g0
  # keys are per-batch example, not number of prior examples
  keys_bxsx2 = np.reshape(random.split(keys[1], B * S), (B, S, 2))
  kl_loss_b = \
      batch_kl_sample_gmm_pmap(keys_bxsx2, post_mean_bxz, post_logvar_bxz,
                               params['gmm_resp'], params['gmm_mean'],
                               params['gmm_logvar'], lfads_hps['var_min'])

  kl_loss_prescale = np.mean(kl_loss_b)#, axis=0)  # IndexError: cannot do a non-empty take from an empty axes.
  kl_loss = kl_scale * kl_loss_prescale

  # Log-likelihood of data given latents.
  log_p_xgz = np.sum(poisson_log_likelihood(x_bxt, lfads['lograte_t'])) / float(B)

  # Implements the idea that inputs to the generator should be minimal, in the
  # sense of attempting to interpret the inferred inputs as actual inputs to a
  # recurrent system, under an assumption of minimal intervention to that
  # system. If all we want to do is keep the mean the at at zero, that's a
  # different loss
  _, ii_prior_mean_cxtxi = batch_decompose_sample(lfads_hps, params['gmm_mean'])
  ii_l2_loss = lfads_hps['ii_l2_reg'] * np.sum(ii_prior_mean_cxtxi**2) / float(C)

  # L2
  l2reg = lfads_hps['l2reg']
  l2_ignore = []
  l2_params = [p for k, p in params.items() if k not in l2_ignore]
  l2_loss = l2reg * optimizers.l2_norm(l2_params)**2

  loss = -log_p_xgz + kl_loss + l2_loss + ii_l2_loss
  all_losses = {'total' : loss, 'nlog_p_xgz' : -log_p_xgz,
                'kl' : kl_loss, 'kl_prescale' : kl_loss_prescale,
                'ii_l2' : ii_l2_loss, 'l2' : l2_loss}

  return all_losses


def lfads_training_loss(params, lfads_hps, key, x_bxt, kl_scale, keep_rate):
  """Pull out the total loss for training.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    kl_scale: scale on KL
    keep_rate: dropout keep rate

  Returns:
    return the total loss for optimization
  """
  losses = lfads_losses(params, lfads_hps, key, x_bxt, kl_scale, keep_rate)
  return losses['total']


def posterior_sample_and_average(params, lfads_hps, key, x_txd):
  """Get the denoised lfad inferred values by posterior sample and average.

  Arguments:
    params: dictionary of lfads parameters
    lfads_hps: dict of LFADS hyperparameters
    key: JAX random state
    x_txd: 2d np.array time by dim trial to denoise

  Returns:
    LFADS dictionary of inferred values, averaged over randomness.
  """
  batch_size = lfads_hps['batch_size']
  keys = random.split(key, batch_size)
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  keep_rate = 1.0
  lfads_dict = batch_lfads(params, lfads_hps, keys, x_bxtxd, keep_rate)
  return average_lfads_batch(lfads_dict)


def posterior_sample(params, lfads_hps, key, x_txd):
  """Get the denoised lfad inferred values by posterior sample and average.

  Arguments:
    params: dictionary of lfads parameters
    lfads_hps: dict of LFADS hyperparameters
    key: JAX random state
    x_txd: 2d np.array time by dim trial to denoise

  Returns:
    LFADS dictionary of inferred values, averaged over randomness.
  """
  batch_size = lfads_hps['batch_size']
  keys = random.split(key, batch_size)
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  keep_rate = 1.0
  return batch_lfads(params, lfads_hps, keys, x_bxtxd, keep_rate)


def prior_sample(params, lfads_hps, key):
  batch_size = lfads_hps['batch_size']
  keys = random.split(key, batch_size)
  return batch_lfads_prior_sample(params, lfads_hps, keys)


### JIT

# JIT functions are orders of magnitude faster.  The first time you use them,
# they will take a couple of minutes to compile, then the second time you use
# them, they will be blindingly fast.

# The static_argnums is telling JAX to ignore the lfads_hps dictionary,
# which means you'll have to pay attention if you change the params.
# How does one force a recompile?
batch_lfads_jit = jit(batch_lfads, static_argnums=(1,))
lfads_losses_jit = jit(lfads_losses, static_argnums=(1,))
lfads_training_loss_jit = jit(lfads_training_loss, static_argnums=(1,))
posterior_sample_and_average_jit = jit(posterior_sample_and_average, static_argnums=(1,))
posterior_sample_jit = jit(posterior_sample, static_argnums=(1,))
prior_sample_jit = jit(prior_sample, static_argnums=(1,))
