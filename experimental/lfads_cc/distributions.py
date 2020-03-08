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


"""Distributions used in the LFADS class conditional tutorial."""


from __future__ import print_function


import functools


from jax import jit
from jax import lax
from jax import pmap
from jax import random
from jax import vmap
from jax.nn import log_softmax
from jax.nn import softmax
import jax.numpy as np
from jax.scipy.special import logsumexp


def poisson_log_likelihood(x, log_rate):
  """Compute the log likelihood under Poisson distribution.

    log poisson(k, r) = log(r^k * e^(-r) / k!)
                      = k log(r) - r - log k!
    log poisson(k, r=exp(l)) = k * l - exp(l) - lgamma(k + 1)

  Args:
    x: binned spike count data.
    log_rate: The (log) rate that define the likelihood of the data
      under the LFADS model.
  Returns:
    The log-likelihood of the data under the model (up to a constant factor).
  """
  return x * log_rate - np.exp(log_rate) - lax.lgamma(x + 1.0)


def diag_gaussian_sample(key, mean, logvar, varmin=1e-16):
  """x ~ N(mean, exp(logvar)).

  Args:
    key: random.PRNGKey for random bits
    mean: np array, mean of gaussian
    logvar: np array, log variance of gaussian
    varmin: minimum variance allowed, helps with numerical stability
  Returns:
    np array samples from the gaussian of the same size as mean and logvar
  """
  key, subkey = random.split(key, 2)
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return mean + np.exp(0.5*logvar_wm) * random.normal(subkey, mean.shape)


def diag_gaussian_log_likelihood(z, mean=0.0, logvar=0.0, varmin=1e-16):
  """Log-likelihood under a Gaussian distribution with diagonal covariance.

     Returns the log-likelihood for each dimension.

  Args:
    z: The value to compute the log-likelihood.
    mean: The mean of the Gaussian
    logvar: The log variance of the Gaussian.
    varmin: Minimum variance allowed (numerically useful).
  Returns:
    The log-likelihood under the Gaussian model.
  """
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return (-0.5 * (logvar + np.log(2*np.pi) +
                  np.square((z-mean)/(np.exp(0.5*(logvar_wm))))))


def diag_multidim_gaussian_log_likelihood(z_u, mean_u, logvar_u, varmin):
  """Log-likelihood under a multidim Gaussian with diagonal covariance.

  Args:
    z_u: compute the log-likelihood for this value.
    mean_u: gaussian mean
    logvar_u: gaussian log variance
    varmin: minimum variance (for numerical stability)

  Returns:
    the log-likelihood for the multidim distribution
  """
  return np.sum(diag_gaussian_log_likelihood(z_u, mean_u, logvar_u, varmin),
                axis=0)


def kl_gauss_gauss(q_mean, q_logvar, p_mean, p_logvar, varmin=1e-16):
  """Compute the KL divergence between two diagonal Gaussian distributions.

            KL(q||p) = E_q[log q(z) - log p(z))]
  Args:
    q_mean: mean of q
    q_logvar: logvar of q
    p_mean: mean of p
    p_logvar: logvar of p
    varmin: minimum variance allowed, useful for numerical stability
  Returns:
    np array of KL, computed analytically, same size as q_mean
  """
  q_logvar = np.log(np.exp(q_logvar) + varmin)
  p_logvar = np.log(np.exp(p_logvar) + varmin)
  return (0.5 * (p_logvar - q_logvar + np.exp(q_logvar - p_logvar)
                 + np.square((q_mean - p_mean) / np.exp(0.5 * p_logvar)) - 1.0))


batch_kl_gauss_b_gauss = vmap(kl_gauss_gauss, in_axes=(0, 0, None, None, None))
batch_kl_gauss_b_gauss_b = vmap(kl_gauss_gauss, in_axes=(0, 0, 0, 0, None))
