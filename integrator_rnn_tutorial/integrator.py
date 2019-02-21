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
import utils


def build_input_and_target_pure_integration(input_params, key):
  """Build white noise input and integration targets."""
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
  targets_t = np.cumsum(white_noise_t)
  inputs_tx1 = np.expand_dims(white_noise_t, axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  return inputs_tx1, targets_tx1

# Now batch it and jit.
build_input_and_target = build_input_and_target_pure_integration
build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
build_inputs_and_targets_jit = jit(build_inputs_and_targets,
                                   static_argnums=(0,))


def plot_batch(ntimesteps, input_bxtxu, target_bxtxo=None, output_bxtxo=None,
               errors_bxtxo=None):
  """Plot some white noise / integrated white noise examples."""
  ntoplot = 10
  plt.figure(figsize=(10,7))
  plt.subplot(311)
  plt.plot(input_bxtxu[0:ntoplot,:,0].T)
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Noise')
  plt.subplot(312)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[0:ntoplot,:,0].T);
    plt.xlim([0, ntimesteps-1]);
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("Integration")
  if errors_bxtxo is not None:
    plt.subplot(313)
    plt.plot(errors_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("|Errors|")
  plt.xlabel('Time')


