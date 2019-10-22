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


"""Distributions used in the LFADS GMM tutorial."""


from __future__ import print_function, division, absolute_import


from jax import jit, lax, pmap, random, vmap
from jax.nn import log_softmax, softmax
import jax.numpy as np
from jax.scipy.special import logsumexp


def poisson_log_likelihood(x, log_rate):
  """Compute the log likelihood under Poisson distribution

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
  """x ~ N(mean, exp(logvar))

  Arguments:
    key: random.PRNGKey for random bits
    mean: np array, mean of gaussian
    logvar: np array, log variance of gaussian
    varmin: minimum variance allowed, helps with numerical stability

  Returns:
    np array samples from the gaussian of the same size as mean and logvar"""
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return mean + np.exp(0.5*logvar_wm) * random.normal(key, mean.shape)


def gmm_sample(key, resps_c, means_c, logvar_c, varmin=1e-16):
  """ Sample mixture of gaussians X ~ \sum_c \pi_c N(mean_c, exp(logvar_c))

  Arguments:
    key: random.PRNGKey for random bits
    resps_c: np.array with shape c, responsibilities in the GMM, \pi in
      the above formula is softmax(resps_c).
    means_c: np.array with shape c, means in GMM
    logvar_c: np.array with shape c, log variances in GMM
    varmin: Minimum variance allowed (numerially useful).

  Returns:
    Sample from the mixture model, np.array
  """
  keys = random.split(key, 2)
  # pick gaussian to sample
  u = random.uniform(keys[0])
  cum_resps_c = np.cumsum(softmax(resps_c))
  cidx = np.argmax(u <= cum_resps_c)
  # sample that gaussian
  return diag_gaussian_sample(keys[1], means_c[cidx], logvar_c[cidx], varmin)


batch_gmm_sample = vmap(gmm_sample, in_axes=(0, None, None, None, None))
batch_gmm_sample_jit = jit(batch_gmm_sample)


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
                  np.square((z-mean)/( np.exp(0.5*(logvar_wm))))))


def diag_multidim_gaussian_log_likelihood(z_u, mean_u, logvar_u, varmin):
  """Log-likelhood under a multidimensional Gaussian distribution with diagonal covariance.
   Returns the log-likelihood for the multidim distribution.
  """
  return np.sum(diag_gaussian_log_likelihood(z_u, mean_u, logvar_u, varmin), axis=0)


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


batch_kl_gauss_gauss = vmap(kl_gauss_gauss, in_axes=(0, 0, None, None, None))


gmm_diag_gaussian_log_likelihood = vmap(diag_gaussian_log_likelihood, (None, 0, 0, None))


def kl_sample_gmm(key, q_mean_u, q_logvar_u,
                  gmm_resps_c, gmm_p_mean_cxu, gmm_p_logvar_cxu, varmin):
  """Sample the KL divergence between a gaussian and mixture of gaussians.
            KL(q||p) = E_q[log q(z) - log p(z))]

  In this case q is gaussian, p is gmm, which requires sampling. So we sample
  z ~ q(z) and then compute log q(z) - log p(z).

  Watch the numerics here, varmin needs to be tiny.

  Arguments:
    key: random.PRNGKey for random bits
    q_mean_u: mean from posterior gaussian distribution with dim u
    q_logvar_u: log variance from posterior gaussian distribution with dim u
    gmm_resps_c: np.array with shape c, responsibilities in the GMM, \pi in
      the above formula is softmax(gmm_resps_c).
    gmm_p_mean_cxu: np.array 2D array with shape mixture by dist dim, means in GMM
    gmm_p_logvar_cxu: "", log variances in GMM
    varmin: Minimum variance allowed (numerially useful).

  Returns:
    Single estimate of the KL divergence.
  """

  # Handle case of one gaussian in the mixture with closed form equations.
  if gmm_resps_c.shape[0] == 1:
    return np.sum(kl_gauss_gauss(q_mean_u, q_logvar_u,
                                 gmm_p_mean_cxu[0,:], gmm_p_logvar_cxu[0,:],
                                 varmin))

  # Otherwise sample the KL
  ll = diag_gaussian_log_likelihood
  gmm_ll = gmm_diag_gaussian_log_likelihood
  sample = diag_gaussian_sample
  keys = random.split(key, 2)

  z_u = sample(keys[0], q_mean_u, q_logvar_u, varmin)
  logq_u = ll(z_u, q_mean_u, q_logvar_u, varmin) # over multigauss dim

  assert varmin <= 1e-15, "Very small or you need to know what you are doing."
  llp_each_gaussian_cxu = gmm_ll(z_u, gmm_p_mean_cxu, gmm_p_logvar_cxu, varmin)
  log_normed_resps_cx1 = np.expand_dims(log_softmax(gmm_resps_c), axis=1)
  logp_u = logsumexp(llp_each_gaussian_cxu + log_normed_resps_cx1, axis=0)

  kl_estimate = np.sum(logq_u - logp_u, axis=0)
  return kl_estimate


batch_samples_kl_sample_gmm = vmap(kl_sample_gmm, in_axes=(0, None, None, None, None, None, None))


def kl_samples_gmm(keys_sx2, q_mean_u, q_logvar_u, gmm_resps_c,
                   gmm_p_mean_cxu, gmm_p_logvar_cxu, varmin):
  """Sample KL between gaussian and gaussian mixture many times and average.

  See comments for kl_sample_gmm for full explanation.

  Args:
    keys_sx2: numpy array of random.PRNGKey, one for each sample.

  Returns:
    KL averaged over S samples.
  """
  sample_kl = batch_samples_kl_sample_gmm
  kl_samples = sample_kl(keys_sx2, q_mean_u, q_logvar_u,
                         gmm_resps_c, gmm_p_mean_cxu, gmm_p_logvar_cxu, varmin)
  kl = np.mean(kl_samples)
  return kl


# Returns batch things, so running over (keys_bx..., z_mean_bx..., z_logvar_bx...)
# keys come in with shape (B, S, 2)
batch_kl_sample_gmm = vmap(kl_samples_gmm, in_axes=(0, 0, 0, None, None, None, None))
batch_kl_sample_gmm_jit = jit(batch_kl_sample_gmm)


## TODO
## The following functions using pmap are prototypes and are not yet used.
##
def batch_kl_sample_gmm_pmap_pre(keys_8xbd8xsx2, z_mean_8xbd8xu,
                                 z_logvar_8xbd8xu, resps_c, gmm_z_mean_cxu,
                                 gmm_z_logvar_cxu, varmin):
  """Sample KL between gaussian and mixture of gaussians many times using pmap.

  See comments for kl_sample_gmm for full explanation.

  Args:
    keys_8xbd8xsx2: numpy array of random.PRNGKey, one for each sample. The size is
      (8, batch_size / 8, S, 2) with S the number of samples. This shape is used for
      setting up a pmap call to 8 devices.
    z_mean_8xbd8xu: mean of gaussian, same shape as keys
    z_logvar_8xbd8xu: logvar of gaussian, same shape as keys

  Returns:
    KL averaged over S samples.
  """


  # This fun gets around jax complaining about vmap not having these parameters.
  def batch_kl_sample_gmm2(keys, z_mean, z_logvar, resps,
                           gmm_z_mean, gmm_z_logvar, varmin):
    return batch_kl_sample_gmm(keys, z_mean, z_logvar, resps,
                           gmm_z_mean, gmm_z_logvar, varmin)

  kwargs = {'resps' : resps_c,
            'gmm_z_mean' : gmm_z_mean_cxu,
            'gmm_z_logvar' : gmm_z_logvar_cxu, 'varmin' : varmin}
  batch_samples_kl_sample_pre_pmap = partial(batch_kl_sample_gmm2, **kwargs)

  pmap_samples = pmap(batch_samples_kl_sample_pre_pmap)

  kl_samples_8xbd8 = pmap_samples(keys_8xbd8xsx2, z_mean_8xbd8xu, z_logvar_8xbd8xu)
  kl = np.mean(kl_samples_8xbd8)
  return kl


def batch_kl_sample_gmm_pmap(keys_bxsx2, z_mean_bxu, z_logvar_bxu, resps_c,
                             gmm_z_mean_cxu, gmm_z_logvar_cxu, varmin):

  """Sample KL between gaussian and mixture of gaussians many times using pmap.

  See comments for kl_sample_gmm for full explanation.

  Args:
    keys_bxsx2: numpy array of random.PRNGKey, one for each sample.
    z_mean_bxu: mean of gaussian, shape of batch x z dim
    z_logvar_bxu: logvar of gaussian, shape of batch x z dim
    resps_c : logits for responsibilities of GMM with c gaussians
    gmm_z_mean_cxu: means for GMM, with c gaussians, each with u dim
    gmm_z_logvar_cxu: log variances for GMM, with c gaussians, each with u dim
    varmin: minimal variance

  Returns:
    KL sample averaged of S samples.

  """
  ndevs = 8
  B, S, _ = keys_bxsx2.shape
  keys_8xbd8xsx2 = np.reshape(keys_bxsx2, (ndevs, B // ndevs, S, 2))
  z_mean_8xbd8xu = np.reshape(z_mean_bxu, (ndevs, B // ndevs, -1))
  z_logvar_8xbd8xu = np.reshape(z_logvar_bxu, (ndevs, B // ndevs, -1))

  # Shard the memory, note the type that goes in vs comes out of the pmap'd lambda.
  keys_8xbd8xsx2 = pmap(lambda x : x)(keys_8xbd8xsx2)
  z_mean_8xbd8xu = pmap(lambda x: x)(z_mean_8xbd8xu)
  z_logvar_8xbd8xu = pmap(lambda x: x)(z_logvar_8xbd8xu)

  return batch_kl_sample_gmm_pmap_pre(keys_8xbd8xsx2, z_mean_8xbd8xu, z_logvar_8xbd8xu,
                                      resps_c, gmm_z_mean_cxu,
                                      gmm_z_logvar_cxu, varmin)


batch_kl_sample_gmm_pmap_jit = jit(batch_kl_sample_gmm_pmap)
