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

from __future__ import print_function

import functools


from jax import jit
from jax import lax
from jax import ops
from jax import random
from jax import vmap
from jax.experimental import optimizers
import jax.numpy as np
import lfads_cc.distributions as dists
import lfads_cc.utils as utils


def sigmoid(x):
  return 0.5 * (np.tanh(x / 2.) + 1)


def class_cond_gauss_params(key, num, dim, mean_std=0.3,
                            var_mean=0.2, var_std=0.025):
  """Params for Gaussians that are class conditioned.

  Args:
    key: random.PRNGKey for random bits
    num: number of gaussians in mixture
    dim: dimension of gaussian
    mean_std: standard deviation of random means around 0
    var_mean : mean of variance of each gaussian
    var_std: standard deviation of the mean of each gaussian

  Returns:
    a dictionary of gaussian parameters with 'means' and 'logvars', each with
      shape (nclasses, dim).
  """
  keys = random.split(key, 2)
  mean_params = mean_std * random.normal(keys[0], shape=(num, dim))
  logvar_params = np.log(var_mean * np.ones((num, dim))
                         + var_std * random.normal(keys[1], shape=(num, dim)))
  return {'means': mean_params, 'logvars': logvar_params}


def linear_params(key, o, u, ifactor=1.0):
  """Params for y = w x.

  Args:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  keys = random.split(key, 2)
  ifactor = ifactor / np.sqrt(u)
  return {'w': random.normal(keys[0], (o, u)) * ifactor}


def affine_params(key, o, u, ifactor=1.0):
  """Params for y = w x + b.

  Args:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  keys = random.split(key, 2)
  ifactor = ifactor / np.sqrt(u)
  return {'w': random.normal(keys[0], (o, u)) * ifactor,
          'b': np.zeros((o,))}


def gru_params(key, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
  """Generate GRU parameters.

  Args:
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

  wRUH = random.normal(keys[0], (n+n, n)) * hfactor
  wRUX = random.normal(keys[1], (n+n, u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)

  wCH = random.normal(keys[2], (n, n)) * hfactor
  wCX = random.normal(keys[3], (n, u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  return {'h0': random.normal(keys[4], (n,)) * hscale,
          'wRUHX': wRUHX,
          'wCHX': wCHX,
          'bRU': np.zeros((n+n,)),
          'bC': np.zeros((n,))}


def affine(params, x):
  """Implement y = w x + b.

  Args:
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
  """Implement y = hat{w} x, where hat{w}_ij = w_ij / |w_{i:}|, norm over j.

  Args:
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

  Args:
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

  Args:
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

  Args:
    params: dictionary of GRU parameters
    h: np array of  hidden state
    x: np array of input

  Returns:
    np array of hidden state after GRU update
  """

  bfg = 0.5
  hx = np.concatenate([h, x], axis=0)
  ru = np.dot(params['wRUHX'], hx) + params['bRU']
  r, u = np.split(ru, 2, axis=0)
  u = u + bfg
  r = sigmoid(r)
  u = sigmoid(u)
  rhx = np.concatenate([r * h, x])
  c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'])
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

  Args:
    rnn_for_scan: function for running RNN one step (h, x) -> (h, h)
      The params already embedded in the function.
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn

  Returns:
    np array of rnn applied to time data with leading dim being time
  """
  _, h_t = lax.scan(rnn_for_scan, h0, x_t)
  return h_t


def run_bidirectional_rnn(params, fwd_rnn, bwd_rnn, x_t):
  """Run an RNN encoder backwards and forwards over some time series data.

  Args:
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
  enc_ends = np.squeeze(np.concatenate([bwd_enc_t[0:1],
                                        fwd_enc_t[-1:]], axis=1), axis=0)
  return full_enc, enc_ends


def init_params(key, hps):
  """Instantiate random LFADS parameters.

  Args:
    key: random.PRNGKey for random bits
    hps: a dict of LFADS hyperparameters

  Returns:
    a dictionary of LFADS parameters
  """
  keys = random.split(key, 11)

  data_dim = hps['data_dim']
  ntimesteps = hps['ntimesteps']
  nclasses = hps['nclasses']
  enc_dim = hps['enc_dim']
  con_dim = hps['con_dim']
  ii_dim = hps['ii_dim']
  gen_dim = hps['gen_dim']
  factors_dim = hps['factors_dim']
  ic_dim = factors_dim
  ib_dim = hps['ib_dim']  # inferred bias is a static input to generator
  z_dim = ib_dim + ic_dim + ntimesteps * ii_dim

  ic_enc_params = {'fwd_rnn': gru_params(keys[0], enc_dim, data_dim + nclasses),
                   'bwd_rnn': gru_params(keys[1], enc_dim, data_dim + nclasses)}

  post_ib_params = affine_params(keys[2], 2*ib_dim, 2*enc_dim)  # m, v <- bi
  post_ic_params = affine_params(keys[3], 2*ic_dim, 2*enc_dim)  # m, v <- bi
  post_ic2gen_params = affine_params(keys[4], gen_dim, ic_dim)
  prior_params = class_cond_gauss_params(keys[5], nclasses, z_dim,
                                         mean_std=0.0, var_mean=0.1,
                                         var_std=0.0)

  con_params = gru_params(keys[6], con_dim, 2*enc_dim + factors_dim + ii_dim)
  con_out_params = affine_params(keys[7], 2*ii_dim, con_dim)  # m, v
  gen_params = gru_params(keys[8], gen_dim, ii_dim + nclasses)
  factors_params = linear_params(keys[9], factors_dim, gen_dim)
  lograte_params = affine_params(keys[10], data_dim, factors_dim)

  return {'con': con_params, 'con_out': con_out_params,
          'factors': factors_params,
          'f0': np.zeros((hps['factors_dim'],)),
          'gen': gen_params,
          'ic_enc': ic_enc_params,
          'ii0': np.zeros((hps['ii_dim'],)),
          'logrates': lograte_params,
          'prior': prior_params,
          'post_ib': post_ib_params,
          'post_ic': post_ic_params,
          'post_ic2gen': post_ic2gen_params}


def one_hot(dim, col_id):
  """Create a one-hot vector with dimension dim and 1 at col_id.

  Args:
    dim: dimension of vector
    col_id: column to under a 1.
  Returns:
    a np.array one-hot vector. If col_id == -1, return a zeros vector.
  """
  return np.where(col_id == -1,
                  np.zeros((dim,)),
                  ops.index_update(np.zeros((dim,)), col_id, 1.0))


def encode(params, hps, key, x_txd, keep_rate, class_id=-1, use_mean=False):
  """Run the LFADS network from input to latent variables.

  Encode the LFADS latent variables. This means running a backwards
  and forwards RNN, then using those RNN states to map to latents for
  the generator initial condition and generator bias. Return also the
  bidirectional encoding for the controller's generation of the
  inferred inputs in the decoding phase.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_txd: np array input for lfads with leading dimension being time
    keep_rate: dropout keep rate
    class_id: The class id of the example.
    use_mean: Use the mean of the posteror dist, not a sample.

  Returns:
    A dict of np arrays: generator initial condition mean, log
    variance, and sample, the inferred bias mean, log variance and
    sample, and also bidirectional encoding of x_txd.

  """
  keys = random.split(key, 5)
  x_txd = run_dropout(x_txd, keys[0], keep_rate)
  class_one_hot = one_hot(hps['nclasses'], class_id)
  class_one_hot_txc = np.tile(class_one_hot, (hps['ntimesteps'], 1))
  enc_input_t = np.concatenate((x_txd, class_one_hot_txc), axis=1)

  con_ins_t, gen_pre_ibs_ics = run_bidirectional_rnn(params['ic_enc'],
                                                     gru, gru, enc_input_t)
  xenc_t = dropout(con_ins_t, keys[1], keep_rate)
  gen_pre_ibs_ics = dropout(gen_pre_ibs_ics, keys[2], keep_rate)

  ib_gauss_params = affine(params['post_ib'], gen_pre_ibs_ics)
  ib_mean, ib_logvar = np.split(ib_gauss_params, 2, axis=0)
  ib_sample = dists.diag_gaussian_sample(keys[3], ib_mean, ib_logvar,
                                         hps['var_min'])
  ib = np.where(use_mean, ib_mean, ib_sample)
  ib = np.where(hps['do_tanh_latents'], np.tanh(ib), ib)

  ic_gauss_params = affine(params['post_ic'], gen_pre_ibs_ics)
  ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
  g0pre_sample = dists.diag_gaussian_sample(keys[4], ic_mean, ic_logvar,
                                            hps['var_min'])
  g0pre = np.where(use_mean, ic_mean, g0pre_sample)
  g0 = affine(params['post_ic2gen'], g0pre)
  g0 = np.where(hps['do_tanh_latents'], np.tanh(g0), g0)

  return {'g0': g0, 'ib': ib, 'ib_logvar': ib_logvar, 'ib_mean': ib_mean,
          'ic_logvar': ic_logvar, 'ic_mean': ic_mean, 'xenc_t': xenc_t}


def decode_one_step(params, hps, keep_rate, use_mean, state, inputs):
  """Run the LFADS network from latent variables to log rates one time step.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    keep_rate: dropout keep rate
    use_mean: Use the mean of the posteror dist, not a sample.
    state: dict of state variables for the decoder RNN
      (controller, inferred input sample, generator, factors)
    inputs: dict of inputs to decoder RNN (keys, inferred bias and data
      encoding).

  Returns:
    A dict of decode values at time t,
      (controller hidden state, inferred input (ii) sample,
       ii mean, ii log var, log rates, factors,  generator hidden state,
       factors, log rates)
  """
  key = inputs['keys_t']
  ccb = inputs['ccb_t']
  ib = inputs['ib_t']
  xenc = inputs['xenc_t']
  c = state['c']
  f = state['f']
  g = state['g']
  # A bit weird but the 'inferred input' is actually state because
  # samples are generated during the decoding pass.
  ii = state['ii']

  keys = random.split(key, 2)
  cin = np.concatenate([xenc, f, ii], axis=0)
  c = gru(params['con'], c, cin)
  cout = affine(params['con_out'], c)
  ii_mean, ii_logvar = np.split(cout, 2, axis=0)  # inferred input params
  ii_sample = dists.diag_gaussian_sample(keys[0], ii_mean, ii_logvar,
                                         hps['var_min'])
  ii = np.where(use_mean, ii_mean, ii_sample)
  ii = np.where(hps['do_tanh_latents'], np.tanh(ii), ii)
  g = gru(params['gen'], g, np.concatenate([ii, ib, ccb], axis=0))
  g = dropout(g, keys[1], keep_rate)
  f = normed_linear(params['factors'], g)
  lograte = affine(params['logrates'], f)
  return {'c': c, 'ccb': ccb, 'g': g, 'f': f, 'ib': ib, 'ii': ii,
          'ii_mean': ii_mean, 'ii_logvar': ii_logvar,
          'lograte': lograte}


def decode_prior_one_step(params, hps, state, inputs):
  """Run the LFADS network from latent variables to log rates one time step.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    state: a dict of state variables, in this case g_{t-1}.
    inputs: a dict of inputs, the random keys, inferred inputs and inferred bias
  Returns:
    A dict with the decodes of the forward pass from the prior sample.
  """
  ccb = inputs['ccb_t']
  ii = inputs['ii_t']
  ib = inputs['ib_t']
  g = state['g']
  g = gru(params['gen'], g, np.concatenate([ii, ib, ccb], axis=0))
  f = normed_linear(params['factors'], g)
  lograte = affine(params['logrates'], f)
  return {'ccb': ccb, 'f': f, 'g': g, 'ib': ib, 'ii': ii, 'lograte': lograte}


def decode_one_step_scan(params, hps, keep_rate, use_mean, state, inputs):
  """Run the LFADS network one step, prepare the inputs and outputs for scan.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    keep_rate: dropout keep rate
    use_mean: Use the mean of the posteror dist, not a sample.
    state: dictionary of LFADS decoding state variables
    inputs : dictionary of LFADS inputs to drive the decoder RNN

  Returns:
    2-tuple of dictionaries, one for the state at time t, the other is
    the decodes at time t.
  """
  decodes = decode_one_step(params, hps, keep_rate, use_mean, state, inputs)
  state = {'c': decodes['c'], 'f': decodes['f'], 'g': decodes['g'],
           'ii': decodes['ii']}
  return state, decodes


def decode_prior_one_step_scan(params, hps, state, inputs):
  """Run the LFADS network one step, prepare the inputs and outputs for scan.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    state: dict of state variables
    inputs: dict of inputs variables.

  Returns:
    2-tuple of state and decodes.
  """
  decodes = decode_prior_one_step(params, hps, state, inputs)
  state = {'g': decodes['g']}
  return state, decodes


def decode(params, hps, key, keep_rate, encodes, class_id=-1, use_mean=False):
  """Run the LFADS network from latent variables to log rates.

    Since the factors (and inferred input) feed back to the controller,
      factors_{t-1} -> controller_t -> ii_t -> generator_t -> factors_t
      is really one big loop and therefor one RNN.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    keep_rate: dropout keep rate
    encodes: dictionary of variables from lfads encoding, including
      (ib_mean, ib_logvar, ic_mean, ic_logvar, xenc_t)
    class_id: int, indicating one-hot encoding for class conditional bias
    use_mean: Use the mean of the posteror dist, not a sample.

  Returns:
    dictionary of lfads decoding variables, including:
      (controller state, generator state, factors, inferred input,
       inferred bias, inferred input mean, inferred input log var,
       log rates)
  """
  keys = random.split(key, 2)

  ii0 = params['ii0']
  ii0 = np.where(hps['do_tanh_latents'], np.tanh(ii0), ii0)
  c0 = params['con']['h0']

  # All the randomness for all T steps at once for efficiency.
  xenc_t = encodes['xenc_t']
  T = xenc_t.shape[0]
  keys_t = random.split(keys[0], T)
  ib_t = np.tile(encodes['ib'], (T, 1))  # as time-dependent input in decoding.

  class_one_hot = one_hot(hps['nclasses'], class_id)
  class_one_hot_txc = np.tile(class_one_hot, (hps['ntimesteps'], 1))

  inputs = {'ccb_t': class_one_hot_txc, 'ib_t': ib_t, 'keys_t': keys_t,
            'xenc_t': xenc_t}
  # A bit weird but the 'inferred input' is actually state because
  # samples are generated during the decoding pass.
  state0 = {'c': c0, 'f': params['f0'], 'g': encodes['g0'], 'ii': ii0}
  decoder = functools.partial(decode_one_step_scan,
                              *(params, hps, keep_rate, use_mean))
  _, decodes = lax.scan(decoder, state0, inputs)
  return decodes


def compose_latent(hps, ib_k, ic_j, ii_txi):
  """Compose latent code from initial condition and inferred input.

  Args:
    hps: dict of LFADS hyperparameters
    ib_k : np.array, inferred bias
    ic_j : np.array, inferred initial condition (also called g0)
    ii_txi: 2D np.array, time by inferred input dim, inferred inputs

  Returns:
    1D np.array of all latents as a single vector
  """
  ii_ti = np.reshape(ii_txi, (-1,))
  return np.concatenate([ib_k, ic_j, ii_ti], axis=0)


batch_compose_latent = vmap(compose_latent, in_axes=(None, 0, 0, 0))


def decompose_latent(hps, z):
  """Break apart the latent code into inferred bias, IC & inferred input.

  Args:
    hps: dict of lfads hyperparameters
    z: latent variables as 1d numpy array with length full latent dim.

  Returns:
    (3-tuple of np.arrays, (inferred bias, initial condition, inferred inputs).
  """
  ib_dim = hps['ib_dim']
  ic_dim = hps['factors_dim']
  ib_k = z[:ib_dim]
  ic_j = z[ib_dim:(ib_dim+ic_dim)]
  ii_ti = z[(ib_dim+ic_dim):]
  ii_txi = np.reshape(ii_ti, (-1, hps['ii_dim']))
  return ib_k, ic_j, ii_txi


def batch_decompose_latent(hps, z_cxz):
  """Break apart batches of latent codes into inferred bias, IC & inferred input.

  Args:
    hps: dict of lfads hyperparameters
    z_cxz: latent variables as 2d numpy array batch by full latent dim. Note
      that _c is batch dim for posteriors or the number of mixture in the prior.

  Returns:
    (3-tuple of np.arrays, (inferred bias, initial condition, inferred inputs).
  """
  ib_dim = hps['ib_dim']
  ic_dim = hps['factors_dim']
  ii_dim = hps['ii_dim']
  ib_cxk = z_cxz[:, :ib_dim]
  ic_cxj = z_cxz[:, ib_dim:(ib_dim+ic_dim)]
  ii_cxti = z_cxz[:, (ib_dim+ic_dim):]
  if ii_cxti.shape[1] > 0:
    ii_cxtxi = np.reshape(ii_cxti, (ii_cxti.shape[0], -1, ii_dim))
  else:
    ii_cxtxi = np.zeros((hps['nclasses'], hps['ntimesteps'], 0))
  return ib_cxk, ic_cxj, ii_cxtxi


def decode_prior(params, hps, key, prior_sample, class_id):
  """Run the LFADS network from latent variables to log rates.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    prior_sample: dict of samples of the latents, inferred input, inferred bias,
      and initial condition.
    class_id: class id to sample from

  Returns:
    A dict of decodes, in this case the samples from the prior.
  """

  ii0 = params['ii0']
  ii0 = np.where(hps['do_tanh_latents'], np.tanh(ii0), ii0)
  ii_txi = prior_sample['ii_t']
  T = ii_txi.shape[0]
  ib_t = np.tile(prior_sample['ib'], (T, 1))

  nclasses = hps['nclasses']
  class_one_hot = one_hot(nclasses, class_id)
  ccb_input = class_one_hot
  ccb_t = np.tile(ccb_input, (T, 1))

  inputs = {'ccb_t': ccb_t, 'ib_t': ib_t, 'ii_t': ii_txi}
  state0 = {'g': prior_sample['g0']}
  decoder = functools.partial(decode_prior_one_step_scan, *(params, hps))
  _, samples = lax.scan(decoder, state0, inputs)
  return samples


def forward_pass(params, hps, key, x_t, class_id, keep_rate, use_mean):
  """Run the LFADS network from input to output.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_t: np array of input with leading dim being time
    class_id: int, indicating one-hot encoding for class conditional bias
    keep_rate: dropout keep rate
    use_mean: Use the mean of the posteror dist, not a sample.

  Returns:
    A dictionary of np arrays of all LFADS values of interest.
  """

  keys = random.split(key, num=2)
  encodes = encode(params, hps, keys[0], x_t, keep_rate, class_id, use_mean)
  decodes = decode(params, hps, keys[1], keep_rate, encodes, class_id, use_mean)

  if hps['train_on'] == 'spike_counts':
    rates = np.exp(decodes['lograte'])
  elif hps['train_on'] == 'continuous':
    rates = decodes['lograte']
  else:
    raise NotImplementedError

  # Saving everything and renaming according to a time axis, making a note this
  # comes from the posterior distribution.
  return {'c_t': decodes['c'],
          'ccb_t': decodes['ccb'],
          'factor_t': decodes['f'],
          'g0': encodes['g0'],
          'gen_t': decodes['g'],
          'ib_post_logvar': encodes['ib_logvar'],
          'ib_post_mean': encodes['ib_mean'],
          'ib': encodes['ib'],
          'ib_t': decodes['ib'],
          'ic_post_logvar': encodes['ic_logvar'],
          'ic_post_mean': encodes['ic_mean'],
          'ii_post_logvar_t': decodes['ii_logvar'],
          'ii_post_mean_t': decodes['ii_mean'],
          'ii_t': decodes['ii'],
          'lograte_t': decodes['lograte'],
          'rate_t': rates,
          'xenc_t': encodes['xenc_t']}


def forward_pass_prior(params, hps, key, class_id):
  """Run the LFADS network from prior samples to log rates.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    class_id: class id to sample from

  Returns:
    A dictionary of np arrays of all LFADS values of interest. Note this is a
    reduced set compared to running lfads with an encoder because we sample the
    prior and then runs only the generative model.
  """
  keys = random.split(key, num=2)
  prior_mean = params['prior']['means'][class_id]
  prior_logvar = params['prior']['logvars'][class_id]
  z_sample = dists.diag_gaussian_sample(keys[0], prior_mean, prior_logvar,
                                        1e-16)
  ib, g0_pre, ii_txi = decompose_latent(hps, z_sample)
  g0 = affine(params['post_ic2gen'], g0_pre)
  g0 = np.where(hps['do_tanh_latents'], np.tanh(g0), g0)
  ii_txi = np.where(hps['do_tanh_latents'], np.tanh(ii_txi), ii_txi)
  ib = np.where(hps['do_tanh_latents'], np.tanh(ib), ib)

  prior_sample = {'g0': g0, 'ib': ib, 'ii_t': ii_txi}
  decodes = decode_prior(params, hps, keys[1], prior_sample, class_id)

  if hps['train_on'] == 'spike_counts':
    rates = np.exp(decodes['lograte'])
  elif hps['train_on'] == 'continuous':
    rates = decodes['lograte']
  else:
    raise NotImplementedError

  return {'ccb_t': decodes['ccb'],
          'factor_t': decodes['f'],
          'g0': g0, 'gen_t': decodes['g'],
          'ib': decodes['ib'],
          'ib_t': decodes['ib'],
          'ii_t': ii_txi,
          'lograte_t': decodes['lograte'],
          'rate_t': rates}


encode_jit = jit(encode)
decode_jit = jit(decode, static_argnums=(1,))
forward_pass_jit = jit(forward_pass, static_argnums=(1,))


# Batching accomplished by vectorized mapping. We simultaneously map over random
# keys for forward-pass randomness and inputs for batching.
batch_forward_pass = vmap(forward_pass,
                          in_axes=(None, None, 0, 0, 0, None, None))


# These shenanigans are thanks to vmap complaining about the decompose_latent
batch_forward_pass_prior = \
    lambda params, hps, keys, class_ids: vmap(lambda key, class_id: forward_pass_prior(params, hps, key, class_id), in_axes=(0,0))(keys, class_ids)


def losses(params, hps, key, x_bxt, class_id_b, kl_scale, keep_rate):
  """Compute the training loss of the LFADS autoencoder.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    class_id_b: class ids, np array of integers for what classes x_bxt are in
    kl_scale: scale on KL
    keep_rate: dropout keep rate

  Returns:
    a dictionary of all losses, including the key 'total' used for optimization
  """

  B = hps['batch_size']
  I = hps['ii_dim']
  T = hps['ntimesteps']

  keys = random.split(key, 2)
  keys_b = random.split(keys[0], B)
  use_mean = False
  lfads = batch_forward_pass(params, hps, keys_b, x_bxt, class_id_b, keep_rate,
                             use_mean)

  post_mean_bxz = batch_compose_latent(hps, lfads['ib_post_mean'],
                                       lfads['ic_post_mean'],
                                       lfads['ii_post_mean_t'])
  post_logvar_bxz = batch_compose_latent(hps, lfads['ib_post_logvar'],
                                         lfads['ic_post_logvar'],
                                         lfads['ii_post_logvar_t'])

  prior_mean_bxz = params['prior']['means'][class_id_b]
  prior_logvar_bxz = params['prior']['logvars'][class_id_b]

  # Sum over time and state dims, average over batch.
  # KL - g0
  kl_loss_bxz = \
      dists.batch_kl_gauss_b_gauss_b(post_mean_bxz, post_logvar_bxz,
                                     prior_mean_bxz, prior_logvar_bxz,
                                     hps['var_min'])

  kl_loss_prescale = np.mean(np.sum(kl_loss_bxz, axis=1))
  kl_loss = kl_scale * kl_loss_prescale

  # Log-likelihood of data given latents.
  if hps['train_on'] == 'spike_counts':
    spikes = x_bxt
    log_p_xgz = (np.sum(dists.poisson_log_likelihood(spikes,
                                                     lfads['lograte_t']))
                 / float(B))
  elif hps['train_on'] == 'continuous':
    continuous = x_bxt
    mean = lfads['lograte_t']
    logvar = np.zeros(mean.shape)  # TODO(sussillo): hyperparameter
    log_p_xgz = (np.sum(dists.diag_gaussian_log_likelihood(continuous,
                                                           mean, logvar))
                 / float(B))
  else:
    raise NotImplementedError

  # Implements the idea that inputs to the generator should be minimal, in the
  # sense of attempting to interpret the inferred inputs as actual inputs to a
  # recurrent system, under an assumption of minimal intervention to that
  # system.
  _, _, ii_post_mean_bxtxi = batch_decompose_latent(hps, post_mean_bxz)
  _, _, ii_prior_mean_bxtxi = batch_decompose_latent(hps, prior_mean_bxz)
  ii_l2_loss = hps['ii_l2_reg'] * (np.sum(ii_prior_mean_bxtxi**2) / float(B) +
                                   np.sum(ii_post_mean_bxtxi**2) / float(B))

  # Implements the idea that the average inferred input should be zero.
  if ii_post_mean_bxtxi.shape[2] > 0:
    ii_tavg_loss = (hps['ii_tavg_reg'] *
                    (np.mean(np.mean(ii_prior_mean_bxtxi, axis=1)**2) +
                     np.mean(np.mean(ii_post_mean_bxtxi, axis=1)**2)))
  else:
    ii_tavg_loss = 0.0

  # L2 - TODO(sussillo): exclusion method is not general to pytrees
  l2reg = hps['l2reg']
  l2_ignore = ['prior']
  l2_params = [p for k, p in params.items() if k not in l2_ignore]
  l2_loss = l2reg * optimizers.l2_norm(l2_params)**2

  loss = -log_p_xgz + kl_loss + l2_loss + ii_l2_loss + ii_tavg_loss
  all_losses = {'total': loss, 'nlog_p_xgz': -log_p_xgz,
                'kl': kl_loss, 'kl_prescale': kl_loss_prescale,
                'ii_l2': ii_l2_loss, 'ii_tavg': ii_tavg_loss, 'l2': l2_loss}

  return all_losses


def training_loss(params, hps, key, x_bxt, class_id_b, kl_scale, keep_rate):
  """Pull out the total loss for training.

  Args:
    params: a dictionary of LFADS parameters
    hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    class_id_b: class ids, np array of integers for what classes x_bxt are in
    kl_scale: scale on KL
    keep_rate: dropout keep rate

  Returns:
    return the total loss for optimization
  """
  tl = losses(params, hps, key, x_bxt, class_id_b, kl_scale, keep_rate)['total']
  return tl


def sample_posterior_and_average(params, hps, key, x_txd, class_id,
                                 batch_size=None):
  """Get the denoised lfad inferred values by posterior sample and average.

  Args:
    params: dictionary of lfads parameters
    hps: dict of LFADS hyperparameters
    key: JAX random state
    x_txd: 2d np.array time by dim trial to denoise
    class_id: one-hot enconding of the class of this example
    batch_size: number of samples, if none, use hps batch size

  Returns:
    LFADS dictionary of inferred values, averaged over randomness.
  """
  if batch_size is None:
    batch_size = hps['batch_size']
  keys = random.split(key, batch_size)
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  class_id_b = class_id * np.ones((batch_size,)).astype(np.int32)
  keep_rate = 1.0
  use_mean = False
  lfads_dict = batch_forward_pass(params, hps, keys, x_bxtxd, class_id_b,
                                  keep_rate, use_mean)
  return utils.average_lfads_batch(lfads_dict)


def sample_posterior(params, hps, key, x_bxtxd, class_id_b):
  """Denoised lfads inferred values by posterior sampling for a batch of data.

  Note, there is no averaging after samples in this function.

  Args:
    params: dictionary of lfads parameters
    hps: dict of LFADS hyperparameters
    key: JAX random state
    x_bxtxd: 3d np.array batch time by dim trial to denoise
    class_id_b: one-hot encoding of the class of this example

  Returns:
    LFADS dictionary of inferred values, averaged over randomness.
  """
  batch_size = x_bxtxd.shape[0]
  keys = random.split(key, batch_size)
  keep_rate = 1.0
  use_mean = False
  return batch_forward_pass(params, hps, keys, x_bxtxd, class_id_b, keep_rate,
                            use_mean)


def posterior_from_mean(params, hps, key, x_bxtxd, class_id_b):
  """Denoised lfads inferred values by pushing posterior mean through decoder.

  Note, there is no averaging after samples in this function.

  Args:
    params: dictionary of lfads parameters
    hps: dict of LFADS hyperparameters
    key: JAX random state
    x_bxtxd: 3d np.array batch time by dim trial to denoise
    class_id_b: one-hot encoding of the class of this example

  Returns:
    LFADS dictionary of inferred values, averaged over randomness.
  """
  batch_size = x_bxtxd.shape[0]
  keys = random.split(key, batch_size)
  keep_rate = 1.0
  use_mean = True
  return batch_forward_pass(params, hps, keys, x_bxtxd, class_id_b, keep_rate,
                            use_mean)


def sample_prior(params, hps, key, class_id_b):
  """Sample the LFADS generative process using random draws from the prior.

  Args:
    params: dictionary of lfads parameters
    hps: dict of LFADS hyperparameters
    key: JAX random state
    class_id_b: one-hot encoding of the classes we sample from

  Returns:
    LFADS dictionary of generated values contextualized by class ids.
  """
  batch_size = class_id_b.shape[0]
  keys_b = random.split(key, batch_size)
  return batch_forward_pass_prior(params, hps, keys_b, class_id_b)


def sample_prior_by_class(params, hps, key, class_id_b):
  """Sample the LFADS generative process using random draws from the prior.

  Arrange data with class id as the outer dictionary.

  Args:
    params: dictionary of lfads parameters
    hps: dict of LFADS hyperparameters
    key: JAX random state
    class_id_b: one-hot encoding of the classes we sample from

  Returns:
    LFADS dictionary of generated values contextualized by class ids.
  """
  prior_dict = sample_prior(params, hps, key, class_id_b)

  batch_size = class_id_b.shape[0]
  prior_dicts = []
  for b in range(batch_size):
    prior_dicts.append({})

  # These shenanigans are thanks to vmap complaining about the decompose_latent
  for k in prior_dict:
    for b in range(batch_size):
      prior_dicts[b][k] = prior_dict[k][b]
  return prior_dicts


### JIT

# JIT functions are orders of magnitude faster.  The first time you use them,
# they will take a couple of minutes to compile, then the second time you use
# them, they will be blindingly fast.

# The static_argnums is telling JAX to ignore the hps dictionary,
# which means you'll have to pay attention if you change the params.
# How does one force a recompile?
batch_forward_pass_jit = jit(batch_forward_pass, static_argnums=(1,))
losses_jit = jit(losses, static_argnums=(1,))
training_loss_jit = jit(training_loss, static_argnums=(1,))
sample_posterior_and_average_jit = jit(sample_posterior_and_average,
                                       static_argnums=(1, 5))
sample_posterior_jit = jit(sample_posterior, static_argnums=(1,))
posterior_from_mean_jit = jit(posterior_from_mean, static_argnums=(1,))
sample_prior_jit = jit(sample_prior, static_argnums=(1,))
