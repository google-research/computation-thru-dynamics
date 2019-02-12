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


### LOAD THE DATA
integrator_rnn_data_file = \
     'trained_data_vrnn_pure_int_0.00735_2019-02-06_19_46_04.h5'
data_dt = 1.0/25.0
max_firing_rate = 20            # spikes per second

lfads_dir = '/tmp/lfads/'
data_dir = os.path.join(lfads_dir, 'data/')
output_dir = os.path.join(lfads_dir, 'output/')
figure_dir = os.path.join(lfads_dir, os.path.join(output_dir, 'figures/'))
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

data_path = os.path.join(data_dir, integrator_rnn_data_file)
data_dict = utils.read_file(data_path)

# Data was generated w/ VRNN w/ tanh, thus (data+1) / 2 -> [0,1]
onp_rng = onp.random.RandomState(seed=0) # For here-level numpy
data_bxtxn = utils.spikify_data((data_dict['hiddens'] + 1)/2, onp_rng, data_dt,
                                max_firing_rate=max_firing_rate)
train_data, eval_data = utils.split_data(data_bxtxn, train_fraction=0.9)


key = random.PRNGKey(0)
key, skeys = utils.keygen(key, 10)


### LFADS Hyper parameters
data_dim = train_data.shape[2]  # input to lfads should have dimensions:
ntimesteps = train_data.shape[1] #   (batch_size x ntimesteps x data_dim)
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


#x_t = random.normal(next(skeys), (ntime, data_dim))
x_t = train_data[0]
x_bxt = train_data[0:batch_size]

keep_rate = 0.98

key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
params = lfads.lfads_params(key, lfads_hps)

# RUN LFADS ENCODE:
start_time = time.time()
ic_mean, ic_logvar, xenc_t = lfads.lfads_encode(params, lfads_hps, key,
                                                x_t, keep_rate)
print("Ran LFADS encode")
print("Time: {:0.2f}".format(time.time() - start_time))


lfads_encode_jit = jit(lfads.lfads_encode)
start_time = time.time()
ic_mean, ic_logvar, xenc_t = lfads_encode_jit(params, lfads_hps, key,
                                              x_t, keep_rate)
print("Ran LFADS encode jit")
print("Time: {:0.2f}".format(time.time() - start_time))


# RUN LFADS DECODE:
start_time = time.time()
c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t = \
      lfads.lfads_decode(params, lfads_hps, key, ic_mean, ic_logvar,
                         xenc_t, keep_rate)
print("Ran LFADS decode")
print("Time: {:0.2f}".format(time.time() - start_time))


lfads_decode_jit = jit(lfads.lfads_decode, static_argnums=(1,))
start_time = time.time()
c_t, ii_mean_t, ii_logvar_t, ii_t, gen_t, factor_t, lograte_t = \
      lfads_decode_jit(params, lfads_hps, key, ic_mean, ic_logvar,
                       xenc_t, keep_rate)
print("Ran LFADS decode jit")
print("Time: {:0.2f}".format(time.time() - start_time))


# RUN LFADS
start_time = time.time()
ld = lfads.lfads(params, lfads_hps, key, x_t, keep_rate)
print("Ran LFADS")
print("Time: {:0.2f}".format(time.time() - start_time))

lfads_jit = jit(lfads.lfads, static_argnums=(1,))
start_time = time.time()
ld = lfads_jit(params, lfads_hps, key, x_t, keep_rate) # Used > 30Gb memory.

print("Ran LFADS jitted") # Never got here, I stopped after 15min and 30Gb
print("Time: {:0.2f}".format(time.time() - start_time))




import pdb; pdb.set_trace()
