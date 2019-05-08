# Computation Through Dynamics

This repository contains a number of subprojects related to the
interlinking of computation and dynamics in artificial and biological
neural systems. 

This is not an officially supported Google product.


## Prerequisites

The code is written in Python 2.7.13. You will also need:

* **JAX** version 0.1.27 or greater ([install](https://github.com/google/jax#installation)) -
* **JAX lib** version 0.1.14 or greater (installed with JAX)
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)
* **h5py** ([install](https://pypi.python.org/pypi/h5py))
* **A GPU** -  XLA compiles these examples to CPU *very slowly*, so best to use a GPU for now.


## LFADS - Latent Factor Analysis via Dynamical Systems

LFADS is a tool for inferring dynamics from noisy, high-dimensional observations
of a dynamics system.  It is a sequential auto-encoder with some very particular
bells and whistles.  Here we have released a tutorial version, written in
*Python / Numpy / JAX* intentionally implemented with readabilily, comprehension and
innovation in mind. You may find the full TensorFlow implementation with run manager 
support ([here](https://github.com/lfads)).

The LFADS tutorial uses the integrator RNN example (see below). The LFADS tutorial example attempts to infer the hidden states of the integrator RNN as well as the white noise input to the RNN. One runs the integrator RNN example and then copies the resulting data file, written in /tmp/ to /tmp/LFADS/data/. Edit the name of the data file in run_lfads.py and then run execute run_lfads.py.

The LFADS tutorial is run through this [Jupyter notebook](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/LFADS%20Tutorial.ipynb).

## Integrator RNN - train a Vanilla RNN to integrate white noise.

Integration is a very simple task and highlights how to set up a loop over time,
batch over multiple input/target examples, use just-in-time compilation to speed
the computation up, and take a gradient in *JAX*.  The data from this example is
also used as input for the LFADS tutorial.

This example is run through this [Jupyter notebook](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/Integrator%20RNN%20Tutorial.ipynb). 


## Fixed point finding - train a GRU to make a binary decision and study it via fixed point finding.
 
The goal of this tutorial is to learn about fixed point finding by running the algorithm on a Gated Recurrent Unit (GRU), which is trained to make a binary decision, namely whether the integral of the white noise input is in total positive or negative, outputing either a +1 or a -1 to encode the decision.

Running the fixed point finder on this decision-making GRU will yield:
1. the underlying fixed points
2. the first order taylor series approximations around those fixed points.

Going through this tutorial will exercise the concepts defined in the [Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks](https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00409).

This example is run through this [Jupyter notebook](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/Fixed%20Point%20Finder%20Tutorial.ipynb). 


## FORCE learning in Echostate networks

In Colab, [Train an echostate network (ESN)](https://colab.research.google.com/github/google-research/computation-thru-dynamics/blob/master/notebooks/FORCE_Learning_in_JAX.ipynb) to generate the chaotic output of another recurrent neural network. This Colab / IPython notebook implements a continuous-time ESN with FORCE learning implemented via recursive least squares (RLS). It also lets you use a GPU and quickly get started with JAX! Two different implementations are explored, one at the JAX / Python level and another at the LAX level. After JIT compilation, the JAX implementation runs very fast.
