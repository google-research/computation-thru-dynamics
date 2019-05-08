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

"""Find the fixed points of a nonlinear system via numerical optimization."""

from __future__ import print_function, division, absolute_import
import datetime

from scipy.spatial.distance import pdist, squareform

import jax.numpy as np
from jax import grad, jacrev, jit, lax, random, vmap
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import os
import time


def find_fixed_points(rnn_fun, candidates, hps, do_print=True):
  """Top-level routine to find fixed points, keeping only valid fixed points.

  This function will:
    Add noise to the fixed point candidates ('noise_var')
    Optimize to find the closest fixed points / slow points (many hps, 
      see optimize_fps)
    Exclude any fixed points whose fixed point loss is above threshold ('fp_tol')
    Exclude any non-unique fixed points according to a tolerance ('unique_tol')
    Exclude any far-away "outlier" fixed points ('outlier_tol')
    
  This top level function runs at the CPU level, while the actual JAX optimization 
  for finding fixed points is dispatched to device.

  Arguments: 
    rnn_fun: one-step update function as a function of hidden state
    candidates: ndarray with shape npoints x ndims
    hps: dict of hyper parameters for fp optimization, including
      tolerances related to keeping fixed points
  
  Returns: 
    4-tuple of (kept fixed points sorted with slowest points first, 
      fixed point losses, indicies of kept fixed points, details of 
      optimization)"""

  npoints, dim = candidates.shape
  
  noise_var = hps['noise_var']
  if do_print and noise_var > 0.0:
    print("Adding noise to fixed point candidates.")
    candidates += onp.random.randn(npoints, dim) * onp.sqrt(noise_var)
    
  if do_print:
    print("Optimizing to find fixed points.")
  fps, opt_details = optimize_fps(rnn_fun, candidates, hps, do_print)

  if do_print and hps['fp_tol'] < onp.inf:  
    print("Excluding fixed points with squared speed above tolerance {:0.5f}.".format(hps['fp_tol']))
  fps, fp_kidxs = fixed_points_with_tolerance(rnn_fun, fps, hps['fp_tol'],
                                              do_print)
  if len(fp_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details
  
  if do_print and hps['unique_tol'] > 0.0:  
    print("Excluding non-unique fixed points.")
  fps, unique_kidxs = keep_unique_fixed_points(fps, hps['unique_tol'],
                                               do_print)
  if len(unique_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details
  
  if do_print and hps['outlier_tol'] < onp.inf:  
    print("Excluding outliers.")
  fps, outlier_kidxs = exclude_outliers(fps, hps['outlier_tol'],
                                        'euclidean', do_print) # TODO(sussillo) Make hp?
  if len(outlier_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details

  if do_print:
    print('Sorting fixed points slowest first.')    
  losses = onp.array(get_fp_loss_fun(rnn_fun)(fps))# came back as jax.interpreters.xla.DeviceArray
  sort_idxs = onp.argsort(losses) 
  fps = fps[sort_idxs]
  losses = losses[sort_idxs]
  try:
    keep_idxs = fp_kidxs[unique_kidxs[outlier_kidxs[sort_idxs]]]
  except:
    import pdb; pdb.set_trace()
  return fps, losses, keep_idxs, opt_details


def get_fp_loss_fun(rnn_fun):
  """Return the per-example mean-squared-error fixed point loss.

  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1

  Returns: function that computes the loss for each example
  """
  batch_rnn_fun = vmap(rnn_fun, in_axes=(0,))
  return jit(lambda h : np.mean((h - batch_rnn_fun(h))**2, axis=1))


def get_total_fp_loss_fun(rnn_fun):
  """Return the MSE fixed point loss averaged across examples.

  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1

  Returns: function that computes the average loss over all examples.
  """
  fp_loss_fun = get_fp_loss_fun(rnn_fun)
  return jit(lambda h : np.mean(fp_loss_fun(h)))


def optimize_fp_core(batch_idx_start, num_batches, update_fun, opt_state):
  """Gradient updates to fixed points candidates in order to find fixed points.

  Uses lax.fori_loop instead of a Python loop to reduce JAX overhead. This 
    loop will be jit'd and run on device.

  Arguments:
    batch_idx_start: Where are we in the total number of batches
    num_batches: how many batches to run
    update_fun: the function that changes params based on grad of loss
    opt_state: the jax optimizer state, containing params and opt state

  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  def run_update(batch_idx, opt_state):
    opt_state = update_fun(batch_idx, opt_state)
    return opt_state

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  return lax.fori_loop(lower, upper, run_update, opt_state)


optimize_fp_core_jit = jit(optimize_fp_core, static_argnums=(1, 2, 3))


def optimize_fps(rnn_fun, fp_candidates, hps, do_print=True):
  """Find fixed points of the rnn via optimization.

  This loop is at the cpu non-JAX level.

  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1, for which the fixed point candidates are trained to be 
      fixed points
    fp_candidates: np array with shape (batch size, state dim) of hidden states 
      of RNN to start training for fixed points
    hps: fixed point hyperparameters
    do_print: Print useful information? 

  Returns:
    np array of numerically optimized fixed points"""

  total_fp_loss_fun = get_total_fp_loss_fun(rnn_fun)

  def get_update_fun(opt_update, get_params):
    """Update the parameters using gradient descent.

    Arguments:
      opt_update: a function to update the optimizer state (from jax.optimizers)
      get_params: a function that extract parametrs from the optimizer state

    Returns:
      a 2-tuple (function which updates the parameters according to the 
        optimizer, a dictionary of details of the optimization)
    """
    def update(i, opt_state):
      params = get_params(opt_state)
      grads = grad(total_fp_loss_fun)(params)    
      return opt_update(i, grads, opt_state)

    return update

  # Build some functions used in optimization.
  decay_fun = optimizers.exponential_decay(hps['step_size'],
                                           hps['decay_steps'],
                                           hps['decay_factor'])
  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=hps['adam_b1'],
                                                     b2=hps['adam_b2'],
                                                     eps=hps['adam_eps'])
  opt_state = opt_init(fp_candidates)
  update_fun = get_update_fun(opt_update, get_params)

  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = fp_candidates.shape[0]
  num_batches = hps['num_batches']
  print_every = hps['opt_print_every']
  num_opt_loops = int(num_batches / print_every)
  fps = get_params(opt_state)
  fp_losses = []
  do_stop = False
  for oidx in range(num_opt_loops):
    if do_stop:
      break
    batch_idx_start = oidx * print_every
    start_time = time.time()
    opt_state = optimize_fp_core_jit(batch_idx_start, print_every, update_fun,
                                     opt_state)
    batch_time = time.time() - start_time

    # Training loss
    fps = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    total_fp_loss = total_fp_loss_fun(fps)
    fp_losses.append(total_fp_loss)
    
    # Saving, printing.
    if do_print:
      s = "    Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}, Training loss {:0.5f}"
      print(s.format(batch_idx_start+1, batch_pidx, batch_time,
                     decay_fun(batch_pidx), total_fp_loss))

    if total_fp_loss < hps['fp_opt_stop_tol']:
      do_stop = True
      if do_print:
        print('Stopping as mean training loss {:0.5f} is below tolerance {:0.5f}.'.format(total_fp_loss, hps['fp_opt_stop_tol']))
    optimizer_details = {'fp_losses' : fp_losses}    
  return fps, optimizer_details


def fixed_points_with_tolerance(rnn_fun, fps, tol=onp.inf, do_print=True):
  """Return fixed points with a fixed point loss under a given tolerance.
  
  Arguments: 
    rnn_fun: one-step update function as a function of hidden state
    fps: ndarray with shape npoints x ndims
    tols: loss tolerance over which fixed points are excluded
    do_print: Print useful information? 

  Returns: 
    2-tuple of kept fixed points, along with indicies of kept fixed points
  """
  fp_loss_fun = get_fp_loss_fun(rnn_fun)
  losses = fp_loss_fun(fps)
  lidxs = losses < tol
  keep_idxs = onp.where(lidxs)[0]
  fps_w_tol = fps[lidxs]
  
  if do_print:
    print("    Kept %d/%d fixed points with tolerance under %f." %
          (fps_w_tol.shape[0], fps.shape[0], tol))
  
  return fps_w_tol, keep_idxs
  

def keep_unique_fixed_points(fps, identical_tol=0.0, do_print=True):
  """Get unique fixed points by choosing a representative within tolerance.

  Args:
    fps: numpy array, FxN tensor of F fixed points of N dimension
    identical_tol: float, tolerance for determination of identical fixed points
    do_print: Print useful information? 

  Returns:
    2-tuple of UxN numpy array of U unique fixed points and the kept indices
  """
  keep_idxs = onp.arange(fps.shape[0])
  if identical_tol <= 0.0:
    return fps, keep_idxs
  if fps.shape[0] <= 1:
    return fps, keep_idxs
  
  nfps = fps.shape[0]
  example_idxs = onp.arange(nfps)
  all_drop_idxs = []

  # If point a and point b are within identical_tol of each other, and the
  # a is first in the list, we keep a.
  distances = squareform(pdist(fps, metric="euclidean"))
  for fidx in range(nfps-1):
    distances_f = distances[fidx, fidx+1:]
    drop_idxs = example_idxs[fidx+1:][distances_f <= identical_tol]
    all_drop_idxs += list(drop_idxs)
       
  unique_dropidxs = onp.unique(all_drop_idxs)
  keep_idxs = onp.setdiff1d(example_idxs, unique_dropidxs)
  if keep_idxs.shape[0] > 0:
    unique_fps = fps[keep_idxs, :]
  else:
    unique_fps = onp.array([], dtype=onp.int64)

  if do_print:
    print("    Kept %d/%d unique fixed points with uniqueness tolerance %f." %
          (unique_fps.shape[0], nfps, identical_tol))
    
  return unique_fps, keep_idxs


def exclude_outliers(data, outlier_dist=onp.inf, metric='euclidean', do_print=True):
  """Exclude points whose closest neighbor is further than threshold.

  Args:
    data: ndarray, matrix holding datapoints (num_points x num_features).
    outlier_dist: float, distance to determine outliers.
    metric: str or function, distance metric passed to scipy.spatial.pdist.
        Defaults to "euclidean"
    do_print: Print useful information? 

  Returns:
    2-tuple of (filtered_data: ndarray, matrix holding subset of datapoints,
      keep_idx: ndarray, vector of bools holding indices of kept datapoints).
  """
  if onp.isinf(outlier_dist):
    return data, onp.arange(len(data))
  if data.shape[0] <= 1:
    return data, onp.arange(len(data))

  # Compute pairwise distances between all fixed points.
  distances = squareform(pdist(data, metric=metric))

  # Find second smallest element in each column of the pairwise distance matrix.
  # This corresponds to the closest neighbor for each fixed point.
  closest_neighbor = onp.partition(distances, 1, axis=0)[1]

  # Return data with outliers removed and indices of kept datapoints.
  keep_idx = onp.where(closest_neighbor < outlier_dist)[0]
  data_to_keep = data[keep_idx]

  if do_print:
    print("    Kept %d/%d fixed points with within outlier tolerance %f." %
          (data_to_keep.shape[0], data.shape[0], outlier_dist))
  
  return data_to_keep, keep_idx                              


def compute_jacobians(rnn_fun, points):
  """Compute the jacobians of the rnn_fun at the points.

  This function uses JAX for the jacobian, and is computed on-device.

  Arguments:
    rnn_fun: RNN one step update function for a single hidden state vector
      h_t -> h_t+1
    points: np array npoints x dim, eval jacobian at this point.

  Returns: 
    npoints number of jacobians, np array with shape npoints x dim x dim
  """
  dFdh = jacrev(rnn_fun)
  batch_dFdh = jit(vmap(dFdh, in_axes=(0,)))
  return batch_dFdh(points)


def compute_eigenvalue_decomposition(Ms, sort_by='magnitude',
                                     do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.

  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.

  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = onp.abs
  elif sort_by == 'real':
    sort_fun = onp.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = onp.linalg.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = onp.linalg.pinv(R).T  # as columns      
      L = L[:, indices]
    decomps.append({'evals' : evals[indices], 'R' : R[:, indices],  'L' : L})
  
  return decomps
