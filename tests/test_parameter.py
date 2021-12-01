#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parameter
----------------------------------

Tests Uniform and Normal parameter priors and sampling functions
"""

import unittest

import numpy as np
import scipy.stats

from enterprise.signals.parameter import UniformPrior, UniformSampler
from enterprise.signals.parameter import NormalPrior, NormalSampler


class TestParameter(unittest.TestCase):
    def test_uniform(self):
        """Test Uniform parameter prior and sampler for various combinations of scalar and vector arguments."""

        # scalar
        p_min, p_max = 0.2, 1.1
        x = 0.5

        msg1 = "Enterprise and scipy priors do not match"
        assert UniformPrior(x, p_min, p_max) == scipy.stats.uniform.pdf(x, p_min, p_max - p_min), msg1

        msg2 = "Enterprise samples have wrong value, type, or size"
        x1 = UniformSampler(p_min, p_max)
        assert p_min < x1 < p_max, msg2
        assert type(x1) == float, msg2

        # vector argument
        x = np.array([0.5, 0.1])
        assert np.allclose(UniformPrior(x, p_min, p_max), scipy.stats.uniform.pdf(x, p_min, p_max - p_min)), msg1

        x1 = UniformSampler(p_min, p_max, size=3)
        assert np.all((p_min < x1) & (x1 < p_max)), msg2
        assert x1.shape == (3,), msg2

        # vector bounds
        p_min, p_max = np.array([0.2, 0.3]), np.array([1.1, 1.2])
        assert np.allclose(UniformPrior(x, p_min, p_max), scipy.stats.uniform.pdf(x, p_min, p_max - p_min)), msg1

        x1 = UniformSampler(p_min, p_max)
        assert np.all((p_min < x1) & (x1 < p_max)), msg2
        assert x1.shape == (2,), msg2

        x1 = UniformSampler(p_min, p_max, size=(3, 2))
        assert np.all((p_min < x1) & (x1 < p_max)), msg2
        assert x1.shape == (3, 2), msg2

    def test_normal(self):
        """Test Normal parameter prior and sampler for various combinations of scalar and vector arguments."""

        # scalar
        mu, sigma = 0.1, 2
        x = 0.5

        msg1 = "Enterprise and scipy priors do not match"
        assert NormalPrior(x, mu, sigma) == scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma ** 2), msg1

        msg2 = "Enterprise samples have wrong value, type, or size"
        x1 = NormalSampler(mu, sigma)
        # this should almost never fail
        assert -5 < (x1 - mu) / sigma < 5, msg2

        # vector argument
        x = np.array([-0.2, 0.1, 0.5])

        assert np.allclose(
            NormalPrior(x, mu, sigma), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma ** 2)
        ), msg1

        x1, x2 = (
            NormalSampler(mu, sigma, size=10),
            scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma ** 2, size=10),
        )
        assert x1.shape == x2.shape, msg2

        # vector bounds; note the different semantics from `NormalPrior`,
        # which returns a vector consistently with `UniformPrior`
        mu, sigma = np.array([0.1, 0.15, 0.2]), np.array([2, 1, 2])
        assert np.allclose(
            np.prod(NormalPrior(x, mu, sigma)), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma ** 2)
        ), msg1

        x1, x2 = NormalSampler(mu, sigma), scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma ** 2)
        assert x1.shape == x2.shape, msg2

        # matrix covariance
        sigma = np.array([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]])

        assert np.allclose(NormalPrior(x, mu, sigma), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma)), msg1

        x1, x2 = NormalSampler(mu, sigma), scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma)
        assert x1.shape == x2.shape, msg2

        x1, x2 = NormalSampler(mu, sigma, size=10), scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma, size=10)
        assert x1.shape == x2.shape, msg2
