###

import jax.numpy as np
from jax import grad, jit, random, vmap
import jax.flatten_util as flatten_util
from jax.config import config

import lfads

import numpy as onp
import os
import utils
import time


key = random.PRNGKey(0)
key, skeys = utils.keygen(key, 10)


### LFADS Hyper parameters
data_dim = 100
ntimesteps = 25
batch_size = 128      # batch size during optimization

# LFADS architecture
enc_dim = 32 #64          # encoder dim
con_dim = 32 #64          # contoller dim
ii_dim = 1            # inferred input dim
gen_dim = 40 # 75          # generator dim
factors_dim = 10 #20      # factors dim

# Optimization HPs that percolates into model
l2reg = 0.000002      # amount of l2 on weights

# Prior distribution parameters
ar_mean = 0.0
ar_autocorrelation_tau = 1.0
ar_noise_variance = 0.1
ic_prior_var = 0.1

lfads_hps = {'data_dim' : data_dim, 'ntimesteps' : ntimesteps,
             'enc_dim' : enc_dim, 'con_dim' : con_dim,
             'ic_prior_var' : ic_prior_var, 'ar_mean' : ar_mean,
             'ar_autocorrelation_tau' : ar_autocorrelation_tau,
             'ar_noise_variance' : ar_noise_variance,
             'ii_dim' : ii_dim, 'gen_dim' : gen_dim,
             'factors_dim' : factors_dim,
             'l2reg' : l2reg,
             'batch_size' : batch_size}


x_t = random.normal(key, (ntimesteps, data_dim))
keep_rate = 0.98
params = lfads.lfads_params(key, lfads_hps)

# RUN LFADS ENCODE:
if True:
  start_time = time.time()
  ic_mean, ic_logvar, xenc_t = lfads.lfads_encode(params, lfads_hps, key,
                                                x_t, keep_rate)
  print("Ran LFADS encode")
  print("Time: {:0.2f}".format(time.time() - start_time))


if True:
  lfads_encode_jit = jit(lfads.lfads_encode)
  start_time = time.time()
  ic_mean, ic_logvar, xenc_t = lfads_encode_jit(params, lfads_hps, key,
                                              x_t, keep_rate)
  print("Ran LFADS encode jit (first time)")
  print("Time: {:0.2f}".format(time.time() - start_time))


# RUN LFADS DECODE:
if True:
  start_time = time.time()
  c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t = \
      lfads.lfads_decode(params, lfads_hps, key, ic_mean, ic_logvar,
                         xenc_t, keep_rate)
  print("Ran LFADS decode")
  print("Time: {:0.2f}".format(time.time() - start_time))


if True:
  lfads_decode_jit = jit(lfads.lfads_decode, static_argnums=(1,))
  start_time = time.time()
  c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t = \
      lfads_decode_jit(params, lfads_hps, key, ic_mean, ic_logvar,
                       xenc_t, keep_rate)
  print("Ran LFADS decode jit (first time)")
  print("Time: {:0.2f}".format(time.time() - start_time))


# RUN LFADS
if True:
  start_time = time.time()
  ld = lfads.lfads(params, lfads_hps, key, x_t, keep_rate)
  print("Ran LFADS")
  print("Time: {:0.2f}".format(time.time() - start_time))

if True:
  lfads_jit = jit(lfads.lfads, static_argnums=(1,))
  start_time = time.time()
  ld = lfads_jit(params, lfads_hps, key, x_t, keep_rate)
  print("Ran LFADS jitted (first time)") # 10 minutes and 30Gb
  print("Time: {:0.2f}".format(time.time() - start_time))


# Things that didn't work: Ditching dictionary, tuple -> one output,
#   jitting both encode and decode functions first, and then calling those,
#   replacing data with random values

import pdb; pdb.set_trace()
