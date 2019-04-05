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


from __future__ import print_function, division, absolute_import
import jax.numpy as np
from jax import jit, random, vmap
import jax.flatten_util as flatten_util
import lfads_tutorial.distributions as dists
import lfads_tutorial.utils as utils


def sigmoid(x):
  return 0.5 * (np.tanh(x / 2.) + 1)


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
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor}


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
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor,
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
  key, skeys = utils.keygen(key, 5)
  ifactor = ifactor / np.sqrt(u)
  hfactor = hfactor / np.sqrt(n)

  wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
  wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)

  wCH = random.normal(next(skeys), (n,n)) * hfactor
  wCX = random.normal(next(skeys), (n,u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  return {'h0' : random.normal(next(skeys), (n,)) * hscale,
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
  do_keep = random.bernoulli(key, keep_rate, x.shape)
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


def run_rnn(params, rnn, x_t, h0=None):
  """Run an RNN module forward in time.

  Arguments:
    params: dictionary of RNN parameters
    rnn: function for running RNN one step
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn, which overwrites param h0

  Returns:
    np array of rnn applied to time data with leading dim being time"""
  h = h0 if h0 is not None else params['h0']
  h_t = []
  for x in x_t:
    h = rnn(params, h, x)
    h_t.append(h)
  return np.array(h_t)


def run_bidirectional_rnn(params, fwd_rnn, bwd_rnn, x_t):
  """Run an RNN encoder backwards and forwards over some time series data.

  Arguments:
    params: a dictionary of bidrectional RNN encoder parameters
    fwd_rnn: function for running forward rnn encoding
    bwd_rnn: function for running backward rnn encoding
    x_t: np array data for RNN input with leading dim being time

  Returns:
    tuple of np array concatenated forward, backward encoding, and
      np array of concatenation of [forward_enc(T), backward_enc(1)]
  """
  fwd_enc_t = run_rnn(params['fwd_rnn'], fwd_rnn, x_t)
  bwd_enc_t = np.flipud(run_rnn(params['bwd_rnn'], bwd_rnn, np.flipud(x_t)))
  full_enc = np.concatenate([fwd_enc_t, bwd_enc_t], axis=1)
  enc_ends = np.concatenate([bwd_enc_t[0], fwd_enc_t[-1]], axis=1)
  return full_enc, enc_ends


def lfads_params(key, lfads_hps):
  """Instantiate random LFADS parameters.

  Arguments:
    key: random.PRNGKey for random bits
    lfads_hps: a dict of LFADS hyperparameters

  Returns:
    a dictionary of LFADS parameters
  """
  key, skeys = utils.keygen(key, 10)

  data_dim = lfads_hps['data_dim']
  ntimesteps = lfads_hps['ntimesteps']
  enc_dim = lfads_hps['enc_dim']
  con_dim = lfads_hps['con_dim']
  ii_dim = lfads_hps['ii_dim']
  gen_dim = lfads_hps['gen_dim']
  factors_dim = lfads_hps['factors_dim']

  ic_enc_params = {'fwd_rnn' : gru_params(next(skeys), enc_dim, data_dim),
                   'bwd_rnn' : gru_params(next(skeys), enc_dim, data_dim)}
  gen_ic_params = affine_params(next(skeys), 2*gen_dim, 2*enc_dim) #m,v <- bi
  ic_prior_params = dists.diagonal_gaussian_params(next(skeys), gen_dim, 0.0,
                                                   lfads_hps['ic_prior_var'])
  con_params = gru_params(next(skeys), con_dim, 2*enc_dim + factors_dim)
  con_out_params = affine_params(next(skeys), 2*ii_dim, con_dim) #m,v
  ii_prior_params = dists.ar1_params(next(skeys), ii_dim,
                                     lfads_hps['ar_mean'],
                                     lfads_hps['ar_autocorrelation_tau'],
                                     lfads_hps['ar_noise_variance'])
  gen_params = gru_params(next(skeys), gen_dim, ii_dim)
  factors_params = linear_params(next(skeys), factors_dim, gen_dim)
  lograte_params = affine_params(next(skeys), data_dim, factors_dim)

  return {'ic_enc' : ic_enc_params,
          'gen_ic' : gen_ic_params, 'ic_prior' : ic_prior_params,
          'con' : con_params, 'con_out' : con_out_params,
          'ii_prior' : ii_prior_params,
          'gen' : gen_params, 'factors' : factors_params,
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
  key, skeys = utils.keygen(key, 3)

  # Encode the input
  x_t = run_dropout(x_t, next(skeys), keep_rate)
  con_ins_t, gen_pre_ics = run_bidirectional_rnn(params['ic_enc'], gru, gru,
                                                 x_t)
  # Push through to posterior mean and variance for initial conditions.
  xenc_t = dropout(con_ins_t, next(skeys), keep_rate)
  gen_pre_ics = dropout(gen_pre_ics, next(skeys), keep_rate)
  ic_gauss_params = affine(params['gen_ic'], gen_pre_ics)
  ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
  return ic_mean, ic_logvar, xenc_t


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

  ntime = lfads_hps['ntimesteps']
  key, skeys = utils.keygen(key, 1+2*ntime)

  # Since the factors feed back to the controller,
  #    factors_{t-1} -> controller_t -> sample_t -> generator_t -> factors_t
  # is really one big loop and therefor one RNN.
  c = c0 = params['con']['h0']
  g = g0 = dists.diag_gaussian_sample(next(skeys), ic_mean,
                                      ic_logvar, lfads_hps['var_min'])
  f = f0 = np.zeros((lfads_hps['factors_dim'],))
  c_t = []
  ii_mean_t = []
  ii_logvar_t = []
  ii_t = []
  gen_t = []
  factor_t = []
  for xenc in xenc_t:
    cin = np.concatenate([xenc, f], axis=0)
    c = gru(params['con'], c, cin)
    cout = affine(params['con_out'], c)
    ii_mean, ii_logvar = np.split(cout, 2, axis=0) # inferred input params
    ii = dists.diag_gaussian_sample(next(skeys), ii_mean,
                                    ii_logvar, lfads_hps['var_min'])
    g = gru(params['gen'], g, ii)
    g = dropout(g, next(skeys), keep_rate)
    f = normed_linear(params['factors'], g)
    # Save everything.
    c_t.append(c)
    ii_t.append(ii)
    gen_t.append(g)
    ii_mean_t.append(ii_mean)
    ii_logvar_t.append(ii_logvar)
    factor_t.append(f)

  c_t = np.array(c_t)
  ii_t = np.array(ii_t)
  gen_t = np.array(gen_t)
  ii_mean_t = np.array(ii_mean_t)
  ii_logvar_t = np.array(ii_logvar_t)
  factor_t = np.array(factor_t)
  lograte_t = batch_affine(params['logrates'], factor_t)

  return c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t


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

  key, skeys = utils.keygen(key, 2)

  ic_mean, ic_logvar, xenc_t = \
      lfads_encode(params, lfads_hps, next(skeys), x_t, keep_rate)

  c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t = \
      lfads_decode(params, lfads_hps, next(skeys), ic_mean, ic_logvar,
                   xenc_t, keep_rate)

  # As this is tutorial code, we're passing everything around.
  return {'xenc_t' : xenc_t, 'ic_mean' : ic_mean, 'ic_logvar' : ic_logvar,
          'ii_t' : ii_t, 'c_t' : c_t, 'ii_mean_t' : ii_mean_t,
          'ii_logvar_t' : ii_logvar_t, 'gen_t' : gen_t, 'factor_t' : factor_t,
          'lograte_t' : lograte_t}


lfads_encode_jit = jit(lfads_encode)
lfads_decode_jit = jit(lfads_decode, static_argnums=(1,))


# Batching accomplished by vectorized mapping.
# We simultaneously map over random keys for forward-pass randomness
# and inputs for batching.
batch_lfads = vmap(lfads, in_axes=(None, None, 0, 0, None))


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
  key, skeys = utils.keygen(key, 2)
  keys = random.split(next(skeys), B)
  lfads = batch_lfads(params, lfads_hps, keys, x_bxt, keep_rate)

  # Sum over time and state dims, average over batch.
  # KL - g0
  ic_post_mean_b = lfads['ic_mean']
  ic_post_logvar_b = lfads['ic_logvar']
  kl_loss_g0_b = dists.batch_kl_gauss_gauss(ic_post_mean_b, ic_post_logvar_b,
                                            params['ic_prior'],
                                            lfads_hps['var_min'])
  kl_loss_g0_prescale = np.sum(kl_loss_g0_b) / B  
  kl_loss_g0 = kl_scale * kl_loss_g0_prescale
  
  # KL - Inferred input
  ii_post_mean_bxt = lfads['ii_mean_t']
  ii_post_var_bxt = lfads['ii_logvar_t']
  keys = random.split(next(skeys), B)
  kl_loss_ii_b = dists.batch_kl_gauss_ar1(keys, ii_post_mean_bxt,
                                          ii_post_var_bxt, params['ii_prior'],
                                          lfads_hps['var_min'])
  kl_loss_ii_prescale = np.sum(kl_loss_ii_b) / B
  kl_loss_ii = kl_scale * kl_loss_ii_prescale
  
  # Log-likelihood of data given latents.
  lograte_bxt = lfads['lograte_t']
  log_p_xgz = np.sum(dists.poisson_log_likelihood(x_bxt, lograte_bxt)) / B

  # L2
  l2reg = lfads_hps['l2reg']
  flatten_lfads = lambda params: flatten_util.ravel_pytree(params)[0]
  l2_loss = l2reg * np.sum(flatten_lfads(params)**2)

  loss = -log_p_xgz + kl_loss_g0 + kl_loss_ii + l2_loss
  all_losses = {'total' : loss, 'nlog_p_xgz' : -log_p_xgz,
                'kl_g0' : kl_loss_g0, 'kl_g0_prescale' : kl_loss_g0_prescale,
                'kl_ii' : kl_loss_ii, 'kl_ii_prescale' : kl_loss_ii_prescale,
                'l2' : l2_loss}
  return all_losses


def lfads_training_loss(params, lfads_hps, key, x_bxt, kl_scale, keep_rate):
  """Pull out the total loss for training.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    keep_rate: dropout keep rate
    kl_scale: scale on KL

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
  skeys = random.split(key, batch_size)  
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  lfads_dict = batch_lfads(params, lfads_hps, skeys, x_bxtxd, 1.0)
  return utils.average_lfads_batch(lfads_dict)


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

                  
  
