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
from enterprise.signals.parameter import TruncNormalPrior, TruncNormalSampler
from enterprise.signals.parameter import LinearExpPrior, LinearExpSampler


class TestParameter(unittest.TestCase):
    def test_uniform(self):
        """Test Uniform parameter prior and sampler for various combinations of scalar and vector arguments."""

        # scalar
        p_min, p_max = 0.2, 1.1
        x = 0.5

        msg1 = "Enterprise and scipy priors do not match"
        assert np.allclose(UniformPrior(x, p_min, p_max), scipy.stats.uniform.pdf(x, p_min, p_max - p_min)), msg1

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

    def test_linearexp(self):
        """Test LinearExp parameter prior and sampler."""

        # scalar
        p_min, p_max = 1, 3
        x = 2
        msg1 = "Scalar prior does not match"
        assert np.allclose(LinearExpPrior(x, p_min, p_max), np.log(10) * 10**2 / (10**3 - 10**1)), msg1

        x = LinearExpSampler(p_min, p_max)
        msg1b = "Scalar sampler out of range"
        assert p_min <= x <= p_max, msg1b

        # vector argument
        x = np.array([0, 1.5, 2.5])
        msg2 = "Vector-argument prior does not match"
        assert np.allclose(
            LinearExpPrior(x, p_min, p_max), np.array([0, 10**1.5, 10**2.5]) * np.log(10) / (10**3 - 10**1)
        ), msg2

        x = LinearExpSampler(p_min, p_max, size=10)
        msg2b = "Vector-argument sampler out of range"
        assert np.all((p_min < x) & (x < p_max)), msg2b

        # vector bounds
        p_min, p_max = np.array([0, 1]), np.array([2, 3])
        x = np.array([1, 2])
        msg3 = "Vector-argument+bounds prior does not match"
        assert np.allclose(
            LinearExpPrior(x, p_min, p_max),
            np.array([10**1 / (10**2 - 10**0), 10**2 / (10**3 - 10**1)]) * np.log(10),
        ), msg3

    def test_normal(self):
        """Test Normal parameter prior and sampler for various combinations of scalar and vector arguments."""

        # scalar
        mu, sigma = 0.1, 2
        x = 0.5

        msg1 = "Enterprise and scipy priors do not match"
        assert np.allclose(
            NormalPrior(x, mu, sigma), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma**2)
        ), msg1

        msg2 = "Enterprise samples have wrong value, type, or size"
        x1 = NormalSampler(mu, sigma)
        # this should almost never fail
        assert -5 < (x1 - mu) / sigma < 5, msg2

        # vector argument
        x = np.array([-0.2, 0.1, 0.5])

        assert np.allclose(
            NormalPrior(x, mu, sigma), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma**2)
        ), msg1

        x1, x2 = (
            NormalSampler(mu, sigma, size=10),
            scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma**2, size=10),
        )
        assert x1.shape == x2.shape, msg2

        # vector bounds; note the different semantics from `NormalPrior`,
        # which returns a vector consistently with `UniformPrior`
        mu, sigma = np.array([0.1, 0.15, 0.2]), np.array([2, 1, 2])
        assert np.allclose(
            np.prod(NormalPrior(x, mu, sigma)), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma**2)
        ), msg1

        x1, x2 = NormalSampler(mu, sigma), scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma**2)
        assert x1.shape == x2.shape, msg2

        # matrix covariance
        sigma = np.array([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]])

        assert np.allclose(NormalPrior(x, mu, sigma), scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sigma)), msg1

        x1, x2 = NormalSampler(mu, sigma), scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma)
        assert x1.shape == x2.shape, msg2
    
    def test_truncnormal(self):
        """Test TruncNormal parameter prior and sampler for various combinations of scalar and vector arguments."""

        # scalar
        mu, sigma, pmin, pmax = 0.1, 2, -2, 3
        x = 0.5
        a, b = (pmin - mu) / sigma, (pmax - mu) / sigma

        msg1 = "Enterprise and scipy priors do not match"
        assert np.allclose(
            TruncNormalPrior(x, mu, sigma, pmin, pmax),
            scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        msg2 = "Enterprise samples have wrong value, type, or size"
        x1 = TruncNormalSampler(mu, sigma, pmin, pmax)
        # this should almost never fail
        assert -5 < (x1 - mu) / sigma < 5, msg2

        # vector argument
        x = np.array([-0.2, 0.1, 0.5])

        assert np.allclose(
            TruncNormalPrior(x, mu, sigma, pmin, pmax),
            scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        x1, x2 = (
            TruncNormalSampler(mu, sigma, pmin, pmax, size=10),
            scipy.stats.truncnorm.rvs(a, b, mu, sigma, size=10)
        )
        assert x1.shape == x2.shape, msg2

        # vector bounds; note the different semantics from `NormalPrior`,
        # which returns a vector consistently with `UniformPrior`
        mu, sigma = np.array([0.1, 0.15, 0.2]), np.array([2, 1, 2])
        pmin, pmax = np.array([-2, -2, -3]), np.array([1, 2, 1])
        a, b = (pmin - mu) / sigma, (pmax - mu) / sigma
        assert np.allclose(
            TruncNormalPrior(x, mu, sigma, pmin, pmax),
            scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        x1, x2 = (
            TruncNormalSampler(mu, sigma, pmin, pmax),
            scipy.stats.truncnorm.rvs(a, b, mu, sigma)
        )
        assert x1.shape == x2.shape, msg2
