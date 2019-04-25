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


"""Routines for creating white noise and integrated white noise."""

from __future__ import print_function, division, absolute_import
import jax.numpy as np
from jax import jit, vmap
from jax import random
import matplotlib.pyplot as plt
import fixed_point_finder.utils as utils


def build_input_and_target_binary_decision(input_params, key):
  """Build white noise input and decision targets.

  The decision is whether the white noise input has a perfect integral
  greater than, or less than, 0. Output a +1 or -1, respectively.

  Arguments: 
    inputs_params: tuple of parameters for this decision task
    key: jax random key for making randomness

  Returns:
    3-tuple of inputs, targets, and the target mask, indicating 
      which time points have optimization pressure on them"""
  
  bias_val, stddev_val, T, ntime = input_params
  dt = T/ntime

  # Create the white noise input.
  key, skeys = utils.keygen(key, 2)
  random_sample = random.normal(next(skeys), (1,))[0]
  bias = bias_val * 2.0 * (random_sample - 0.5)
  stddev = stddev_val / np.sqrt(dt)
  random_samples = random.normal(next(skeys), (ntime,))
  noise_t = stddev * random_samples
  white_noise_t = bias + noise_t

  # * dt, intentionally left off to get output scaling in O(1).
  pure_integration_t = np.cumsum(white_noise_t)
  decision = 2.0*((pure_integration_t[-1] > 0.0)-0.5)
  targets_t = np.zeros(pure_integration_t.shape[0]-1)
  targets_t = np.concatenate([targets_t,
                              np.array([decision], dtype=float)], axis=0)
  inputs_tx1 = np.expand_dims(white_noise_t, axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  target_mask = np.array([ntime-1]) # When target is defined.
  return inputs_tx1, targets_tx1, target_mask


# Now batch it and jit.
build_input_and_target = build_input_and_target_binary_decision
build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
build_inputs_and_targets_jit = jit(build_inputs_and_targets,
                                   static_argnums=(0,))


def plot_batch(input_params, input_bxtxu, target_bxtxo=None, output_bxtxo=None,
               errors_bxtxo=None, ntoplot=1):
  """Plot some white noise / integrated white noise examples."""
  bval, sval, T, ntimesteps = input_params
  plt.figure(figsize=(16,12))
  plt.subplot(311)
  xs = np.arange(1, ntimesteps+1)
  plt.plot(xs, input_bxtxu[0:ntoplot,:,0].T)
  plt.xlim([1, ntimesteps])
  plt.ylabel('Noise')
  plt.subplot(312)
  if output_bxtxo is not None:
    plt.plot(xs, output_bxtxo[0:ntoplot,:,0].T);
    plt.xlim([1, ntimesteps]);
  if target_bxtxo is not None:
    plt.stem([ntimesteps]*ntoplot, target_bxtxo[0:ntoplot,ntimesteps-1,0].T, '--');
    plt.xlim([1, ntimesteps]);
    plt.ylabel("Decision")
  if errors_bxtxo is not None:
    plt.subplot(313)
    plt.plot(xs, errors_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([1, ntimesteps]);
    plt.ylabel("|Errors|")
  plt.xlabel('Timesteps')


