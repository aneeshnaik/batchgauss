#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single function: sample, draws batched samples from N multivariate Gaussians.

Created: July 2022
Author: A. P. Naik
"""
import numpy as np


def sample(means, covs, rng=None):
    r"""
    Batch sample from N different M-dimensional Gaussian distributions.

    You have N different Gaussian distributions. Each is multivariate (M
    dimensions), and specified by a mean and an MxM dimensional covariance
    matrix. Together, all the means are specifed by an NxM array and the
    covariances are an NxMxM array. This function draws one sample from each
    Gaussian, and returns all the samples as an NxM array.

    Parameters
    ----------
    means : array, shape (N, M)
        Means.
    covs : array, shape (N, M, M)
        Covariance matrices.
    rng : np.random.Generator, optional
        Random number generator from numpy, e.g. as generated via the function
        np.random.default_rng. If None is given, then the RNG used is
        default_rng(42). The default is None.

    Returns
    -------
    samples : array, shape (N, M)
        Gaussian samples.

    Notes
    -----
    Mathematically, the sampler works by (batch-)making a Cholesky
    decomposition of the covariance matrices :math:`\Sigma = L L^T` and taking
    advantage of the fact that if :math:`X` is a batch of samples from a
    standard normal distribution then :math:`LX + \mu` has covariance
    :math:`\Sigma` and mean :math:`\mu`.

    """

    # get and check shapes
    N = means.shape[0]
    M = means.shape[-1]
    assert means.shape == (N, M)
    assert covs.shape == (N, M, M)

    # RNG
    if rng is None:
        rng = np.random.default_rng(42)

    # cholesky decomposition
    L = np.linalg.cholesky(covs)

    # samples y = Lx + mu
    x = rng.standard_normal((N, M))
    samples = (L @ x[..., None]).squeeze() + means
    return samples
