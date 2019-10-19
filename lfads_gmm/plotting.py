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


"""Plotting functions for LFADS and the data RNN example."""


from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as onp

from scipy import stats
from sklearn.decomposition import PCA

def plot_data_pca(data_dict):
  """Plot the PCA skree plot of the hidden units in the integrator RNN."""
  f = plt.figure()
  ndata, ntime, nhidden = data_dict['hiddens'].shape

  print('Number of data examples: ', ndata)
  print('Number of timesteps: ', ntime)
  print('Number of data dimensions: ', nhidden)
  pca = PCA(n_components=100)
  pca.fit(onp.reshape(data_dict['hiddens'], [ndata * ntime, nhidden]))

  plt.plot(onp.arange(1, 16), onp.cumsum(pca.explained_variance_ratio_)[0:15],
           '-o');
  plt.plot([1, 15], [0.95, 0.95])
  plt.xlabel('PC #')
  plt.ylabel('Cumulative Variance')
  plt.xlim([1, 15])
  plt.ylim([0.3, 1]);
  return f


def plot_data_example(input_bxtxu, hidden_bxtxn=None,
                      output_bxtxo=None, target_bxtxo=None, bidx=None):
  """Plot a single example of the data from the data integrator RNN."""
  if bidx is None:
    bidx = onp.random.randint(0, input_bxtxu.shape[0])
  ntoplot = 10
  ntimesteps = input_bxtxu.shape[1]
  f = plt.figure(figsize=(10,8))
  plt.subplot(221)
  plt.plot(input_bxtxu[bidx,:,0], 'b')
  plt.plot(input_bxtxu[bidx,:,2], 'k')
  plt.plot([0, ntimesteps], [0, 0], color=[0.0, 0.0, 0.0, 0.1])
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Input')
  plt.title('Example %d'%bidx)
  plt.subplot(222)
  plt.plot(input_bxtxu[bidx,:,1], 'b')
  plt.plot(input_bxtxu[bidx,:,3], 'k')
  plt.plot([0, ntimesteps], [0, 0], color=[0.0, 0.0, 0.0, 0.1])  
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Input')
  plt.title('Example %d'%bidx)
  
  if hidden_bxtxn is not None:
    plt.subplot(223)
    plt.plot(hidden_bxtxn[bidx, :, 0:ntoplot] + 0.25*onp.arange(0, ntoplot, 1), 'b')
    plt.ylabel('Hiddens')
    plt.xlim([0, ntimesteps-1]);
    plt.xlabel('Time')
  plt.subplot(224)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[bidx,:,0].T, 'r');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel('Output / Targets')
    plt.xlabel('Time')
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[bidx,:,0], 'k');
    plt.xlim([0, ntimesteps-1]);
  return f


def plot_data_stats(data_dict, data_bxtxn, data_dt, bidx=0):
  """Plot the statistics of the data integrator RNN data after spikifying."""
  print(onp.mean(onp.sum(data_bxtxn, axis=1)), "spikes/second")
  f = plt.figure(figsize=(12,4))
  plt.subplot(141)
  plt.hist(onp.mean(data_bxtxn, axis=1).ravel()/data_dt);
  plt.xlabel('spikes / sec')
  plt.subplot(142)
  plt.imshow(data_dict['hiddens'][bidx,:,:].T)
  plt.xlabel('time')
  plt.ylabel('neuron #')
  plt.title('Sample trial rates')
  plt.subplot(143);
  plt.imshow(data_bxtxn[bidx,:,:].T)
  plt.xlabel('time')
  plt.ylabel('neuron #')
  plt.title('spikes')
  plt.subplot(144)
  plt.stem(onp.mean(onp.sum(data_bxtxn, axis=1), axis=0));
  plt.xlabel('neuron #')
  plt.ylabel('spikes / sec');
  return f


def plot_losses(tlosses, elosses, sampled_every):
  """Plot the losses associated with training LFADS."""
  f = plt.figure(figsize=(15, 12))
  for lidx, k in enumerate(tlosses):
    plt.subplot(3, 2, lidx+1)
    tl = tlosses[k].shape[0]
    x = onp.arange(0, tl) * sampled_every
    plt.plot(x, tlosses[k], 'k')
    plt.plot(x, elosses[k], 'r')
    plt.axis('tight')
    plt.title(k)

  return f


def plot_priors(params):
  """Plot the parameters of the LFADS priors."""
  prior_dicts = {'ic' : params['ic_prior'], 'ii' : params['ii_prior']}
  pidxs = (pidx for pidx in onp.arange(1,12))
  f = plt.figure(figsize=(12,8))
  for k in prior_dicts:
    for j in prior_dicts[k]:
      plt.subplot(2,3,next(pidxs));
      data = prior_dicts[k][j]
      if "log" in j:
        data = onp.exp(data)
        j_title = j.strip('log')
      else:
        j_title = j
      plt.stem(data)
      plt.title(k + ' ' + j_title)
  return f


def remove_outliers(A, nstds=3):
  clip = nstds * onp.std(A)
  A_mean = onp.mean(A)
  A_show = onp.where(A < A_mean - clip, A_mean - clip, A)
  return onp.where(A_show > A_mean + clip, A_mean + clip, A_show)



def plot_lfads(x_txd, lfads_dict, data_dict=None, dd_bidx=None,
               renorm_fun=None):
  """Plot the full state of LFADS operating on a single example."""
  print("bidx: ", dd_bidx)
  ld = lfads_dict

  def remove_outliers(A, nstds=3):
    clip = nstds * onp.std(A)
    A_mean = onp.mean(A)
    A_show = onp.where(A < A_mean - clip, A_mean - clip, A)
    return onp.where(A_show > A_mean + clip, A_mean + clip, A_show)
    
  f = plt.figure(figsize=(12,20))
  if x_txd is not None:         # for plotting prior samples
    plt.subplot(621)
    plt.imshow(x_txd)
    plt.title('x')

  if 'xenc_t' in ld.keys():
    plt.subplot(622)
    x_enc = remove_outliers(ld['xenc_t'])
    plt.imshow(x_enc)
    plt.title('x enc')

  plt.subplot(623)
  gen = remove_outliers(ld['gen_t'])
  plt.imshow(gen)
  plt.title('generator')

  plt.subplot(624)
  factors = remove_outliers(ld['factor_t'])
  plt.imshow(factors)
  plt.title('factors')

  plt.subplot(625)
  rates = remove_outliers(onp.exp(ld['lograte_t']))
  plt.imshow(rates)
  plt.title('rates')    

  if data_dict is not None:
    true_rates = renorm_fun(data_dict['hiddens'][dd_bidx])
    plt.subplot(626)
    plt.imshow(true_rates)
    plt.title('True rates')

  plt.subplot(627)        
  if 'ic_mean' in ld.keys():
    ic = ld['ic_mean']
  else:
    ic = ld['g0']
  plt.stem(ic)
  plt.title('g0 mean')
    

  if 'c_t' in ld.keys():
    plt.subplot(628)    
    con = remove_outliers(ld['c_t'])
    plt.imshow(con)
    plt.title('controller')

  plt.subplot(629)
  if 'ii_mean_t' in ld.keys():
    ii = ld['ii_mean_t']
  else:
    ii = ld['ii_t']
  ntime = ii.shape[0]
  scaler = 1*onp.expand_dims(onp.arange(ii.shape[1]), axis=0)  
  plt.plot(ii + scaler, 'b')
  plt.title('Inferred inputs')
 
  if data_dict is not None:
    plt.subplot(6,2,10)
    ninputs = data_dict['inputs'].shape[2]      
    scaler = onp.expand_dims(onp.arange(ninputs), axis=0)    
    plt.plot(data_dict['inputs'][dd_bidx] + scaler, 'c')
    plt.title('Inputs to data RNN')

  plt.subplot(6,2,11)
  plt.plot(ld['ib_t'], 'k')
  plt.title('Inferred bias')
  
  plt.subplot(6,2,12)
  ntoplot=8
  a = 0.25
  plt.plot(rates[:, :ntoplot] + a*onp.arange(0, ntoplot, 1), 'b')
  if data_dict is not None:
    plt.plot(true_rates[:, 0:ntoplot] + a*onp.arange(0, ntoplot, 1), 'r')
    plt.title('LFADS rates (blue), True rates (red)')
  else:
    plt.title('LFADS rates (blue)')
  plt.xlabel('timesteps')
  
  return f



def plot_lfads_new(x_txd, avg_lfads_dict, figsize=(16,16), nstds=3):
  """Plot the full state ofLFADS operating on a single example."""
  ld = avg_lfads_dict

  def remove_outliers(A, nstds=3):
    clip = nstds * onp.std(A)
    A_mean = onp.mean(A)
    A_show = onp.where(A < A_mean - clip, A_mean - clip, A)
    return onp.where(A_show > A_mean + clip, A_mean + clip, A_show)

  f = plt.figure(figsize=figsize)

  plt.subplot(361)
  x_enc = remove_outliers(ld['xenc_t'], nstds)
  plt.imshow(x_enc.T)
  plt.title('x enc')

  plt.subplot(362)
  gen = remove_outliers(ld['gen_t'], nstds)
  plt.imshow(gen.T)
  plt.title('generator')

  plt.subplot(363)
  factors = remove_outliers(ld['factor_t'], nstds)
  plt.imshow(factors.T)
  plt.title('factors')

  plt.subplot(364)
  rates = remove_outliers(onp.exp(ld['lograte_t']), nstds)
  plt.imshow(rates.T)
  plt.title('rates')

  plt.subplot(365)
  data = remove_outliers(x_txd, nstds)
  plt.imshow(data.T)
  plt.title('x')

  plt.subplot(334)
  ic_mean = ld['ic_mean']
  ic_std = onp.exp(0.5*ld['ic_logvar'])
  plt.stem(ic_mean)
  plt.title('g0 mean')

  plt.subplot(335)
  con = remove_outliers(ld['c_t'], nstds)
  plt.imshow(con.T)
  plt.title('controller')

  plt.subplot(336)
  ii_mean = ld['ii_mean_t']
  plt.plot(ii_mean + onp.expand_dims(onp.arange(ii_mean.shape[1]), axis=0), 'b')
  plt.title('inferred input mean')
  plt.legend(('LFADS inferred input', 'rescaled true input to integrator RNN'))

  plt.subplot(313)
  ntoplot=8
  a = 0.25
  plt.plot(rates[:, 0:ntoplot] + a*onp.arange(0, ntoplot, 1), 'b')
  plt.title('LFADS rates (blue), True rates (red)')
  plt.xlabel('timesteps')

  return f
