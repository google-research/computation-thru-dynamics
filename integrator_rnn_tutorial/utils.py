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


"""Utility functions related reading / writing files and random keys."""

from __future__ import print_function, division, absolute_import
import h5py
import jax.numpy as np
from jax import random
import numpy as onp  # original CPU-backed NumPy
import os

def ensure_dir(file_path):
  """Make sure the directory exists, create if it does not."""
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)


def write_file(data_fname, data_dict):
  """Write a simple dictionary using h5py."""
  try:
    ensure_dir(data_fname)
    with h5py.File(data_fname, 'w') as hf:
      for k in data_dict:
        hf.create_dataset(k, data=data_dict[k])
        # add attributes
  except IOError:
    print("Cannot write % for writing." % data_fname)
    raise


def read_file(data_fname):
  """Read a simple dictionary of np arrays using h5py."""
  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: onp.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print("Cannot open %s for reading." % data_fname)
    raise


def keygen(key, nkeys):
  """Generate randomness that JAX can use by splitting the JAX keys.
  Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator
  Returns:
    2-tuple (new key for further generators, key generator)
  """
  keys = random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])
