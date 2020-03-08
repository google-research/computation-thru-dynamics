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


"""Utility functions for the LFADS tutorial."""


from __future__ import print_function


import os


import h5py
import numpy as onp  # original numpy


MAX_SEED_INT = 10000000


def split_data(data_b, train_fraction):
  """Split the data into training and evaluation sets.

  Args:
    data_b: np array, data whose leading dimension is the batch
    train_fraction: fraction of data used for training
  Returns:
    2-tuple of trainind and eval splits
  """
  train_data_offset = 0
  ndata = data_b.shape[0]
  eval_data_offset = int(train_fraction * ndata)
  train_data = data_b[train_data_offset:eval_data_offset]
  eval_data = data_b[eval_data_offset:]

  return train_data, eval_data


def spikify_data(data_bxtxn, rng, dt=1.0, max_firing_rate=100):
  """Apply spikes to a continuous dataset whose values are between 0 and 1.

  Args:
    data_bxtxn: nexamples x time x dim trials
    rng: random number generator
    dt: how often the data are sampled
    max_firing_rate: the firing rate that is associated with a value of 1.0
  Returns:
    spikified_e: a list of length b of the data represented as spikes,
      sampled from the underlying poisson process.
  """
  spikes_e = []
  B, T, N = data_bxtxn.shape
  for data_txn in data_bxtxn:
    spikes = onp.zeros([T, N]).astype(onp.int)
    for n in range(N):
      f = data_txn[:, n]
      s = rng.poisson(f*max_firing_rate*dt, size=T)
      spikes[:, n] = s
    spikes_e.append(spikes)
  return onp.array(spikes_e)


def merge_losses_dicts(list_of_dicts):
  """List of dictionaries converted to dictionary of lists (numpy arrays).

  Args:
    list_of_dicts: list of dictionaries
  Returns:
    A single dictionary with merged keys and values of np arrays.
  """
  merged_d = {}
  d = list_of_dicts[0]
  for k in d:
    merged_d[k] = []
  for d in list_of_dicts:
    for k in d:
      merged_d[k].append(d[k])
  for k in merged_d:
    merged_d[k] = onp.array(merged_d[k])
  return merged_d


def average_lfads_batch(lfads_dict):
  """Average over a run of lfads, for posterior sample and average.

  Args:
    lfads_dict: A dictionary of lfads hidden states, each value is a np array
      with leading dim being batch.
  Returns:
    A dictionary with np arrays whose leading dim has been averaged over.
  """
  avg_dict = {}
  for k in lfads_dict:
    avg_dict[k] = onp.mean(lfads_dict[k], axis=0)
  return avg_dict


def ensure_dir(file_path):
  """Make sure the directory exists, create if it does not."""

  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)


def write_file(data_fname, data_dict):
  """Write a simple dictionary using h5py.

  Args:
    data_fname: name of file to save
    data_dict: dictionary of values to save, each value in dict is a np array
  """
  try:
    ensure_dir(data_fname)

    with h5py.File(data_fname, 'w') as hf:
      for k in data_dict:
        hf.create_dataset(k, data=data_dict[k])
        # add attributes
  except IOError:
    print('Cannot write % for writing.' % data_fname)
    raise


def read_file(data_fname):
  """Read a simple dictionary of np arrays using h5py.

  Args:
    data_fname: name of file to load
  Returns:
    dictionary of np arrays
  """
  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: onp.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print('Cannot open %s for reading.' % data_fname)
    raise
