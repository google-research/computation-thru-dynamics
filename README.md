# Computation Through Dynamics

This repository contains a number of subprojects related to the interlinking of
computation and dynamics in artificial and biological neural systems.

This is not an officially supported Google product.

## LFADS - Latent Factor Analysis via Dynamics Systems

LFADS is a tool for inferring dynamics from noisy, high-dimensional observations
of a dynamics system.  It is a sequential auto-encoder with some very particular
bells and whistles.  Here we have released a tutorial version, written in
Python/Numpy/JAX intentionally implemented with readabilily, comprehension and
innovation in mind.

## Integrator RNN - Train a Vanilla RNN in JAX to integrate white noise.
Integration is a very simple task and highlights how to set up a loop over time,
batch over multiple input/target examples, use just-in-time compilation to speed
the computation up, and take a gradient in JAX.  The data from this example is
also used as input for the LFADS tutorial.
