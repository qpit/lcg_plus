# lcg_plus

Continuous variable circuit simulator with Gaussian components and photon number resolving detectors. This is an extension of the [bosonic backend](https://strawberryfields.ai/photonics/demos/run_intro_bosonic.html) from strawberryfields with improved photon number resolving detector capabilities.

## Requirements
- See `pyproject.toml`

## Installation
`pip install -e .` from root.

## Features
- Simulation of phase-space Wigner functions in the Gaussian representation by tracking covariance matrices, displacement vectors, and weights.
- Non-Gaussian states and measurements are represented as linear combinations of multivariate Gaussians.

Please see our preprint for more information.
>Olga Solodovnikova, Ulrik L. Andersen, Jonas S. Neergaard-Nielsen "Fast simulations of continuous-variable circuits using the coherent state decomposition" [arXiv:2508.06175 \[quant-ph\]](http://arxiv.org/abs/2508.06175)

## Tutorials
See Jupyter notebooks in `demos_final/`.



