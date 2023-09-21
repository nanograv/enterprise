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

from enterprise.signals.parameter import UniformPrior, UniformSampler, Uniform, UniformPPF
from enterprise.signals.parameter import NormalPrior, NormalSampler, Normal, NormalPPF
from enterprise.signals.parameter import TruncNormalPrior, TruncNormalSampler, TruncNormal
from enterprise.signals.parameter import LinearExpPrior, LinearExpSampler, LinearExpPPF


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

        msg3 = "Enterprise and scipy PPF do not match"
        assert np.allclose(UniformPPF(x, p_min, p_max), scipy.stats.uniform.ppf(x, p_min, p_max - p_min)), msg3

        # vector argument
        x = np.array([0.5, 0.1])
        assert np.allclose(UniformPrior(x, p_min, p_max), scipy.stats.uniform.pdf(x, p_min, p_max - p_min)), msg1

        x1 = UniformSampler(p_min, p_max, size=3)
        assert np.all((p_min < x1) & (x1 < p_max)), msg2
        assert x1.shape == (3,), msg2

        # vector argument
        assert np.allclose(UniformPPF(x, p_min, p_max), scipy.stats.uniform.ppf(x, p_min, p_max - p_min)), msg3

        # vector bounds
        p_min, p_max = np.array([0.2, 0.3]), np.array([1.1, 1.2])
        assert np.allclose(UniformPrior(x, p_min, p_max), scipy.stats.uniform.pdf(x, p_min, p_max - p_min)), msg1
        assert np.allclose(UniformPPF(x, p_min, p_max), scipy.stats.uniform.ppf(x, p_min, p_max - p_min)), msg3

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

        msg1c = "Scalar PPF does not match"
        x = 0.5
        assert np.allclose(
            LinearExpPPF(x, p_min, p_max), np.log10(10**p_min + x * (10**p_max - 10**p_min))
        ), msg1c

        # vector argument
        x = np.array([0, 1.5, 2.5])
        msg2 = "Vector-argument prior does not match"
        assert np.allclose(
            LinearExpPrior(x, p_min, p_max), np.array([0, 10**1.5, 10**2.5]) * np.log(10) / (10**3 - 10**1)
        ), msg2

        x = LinearExpSampler(p_min, p_max, size=10)
        msg2b = "Vector-argument sampler out of range"
        assert np.all((p_min < x) & (x < p_max)), msg2b

        x = np.array([0.5, 0.75])
        msg2c = "Vector-argument PPF does not match"
        assert np.allclose(
            LinearExpPPF(x, p_min, p_max), np.log10(10**p_min + x * (10**p_max - 10**p_min))
        ), msg2c

        # vector bounds
        p_min, p_max = np.array([0, 1]), np.array([2, 3])
        x = np.array([1, 2])
        msg3 = "Vector-argument+bounds prior does not match"
        assert np.allclose(
            LinearExpPrior(x, p_min, p_max),
            np.array([10**1 / (10**2 - 10**0), 10**2 / (10**3 - 10**1)]) * np.log(10),
        ), msg3

        # Vector PPF
        x = np.array([0.5, 0.75])
        p_min, p_max = np.array([0, 1]), np.array([2, 3])
        msg3c = "Vector-argument PPF+bounds does not match"
        assert np.allclose(
            LinearExpPPF(x, p_min, p_max), np.log10(10**p_min + x * (10**p_max - 10**p_min))
        ), msg3c

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

        msg3 = "Enterprise and scipy PPF do not match"
        assert np.allclose(NormalPPF(x, mu, sigma), scipy.stats.norm.ppf(x, loc=mu, scale=sigma)), msg3

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

        x = np.array([0.1, 0.25, 0.65])
        assert np.allclose(NormalPPF(x, mu, sigma), scipy.stats.norm.ppf(x, loc=mu, scale=sigma)), msg3

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
            TruncNormalPrior(x, mu, sigma, pmin, pmax), scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        msg2 = "Enterprise samples have wrong value, type, or size"
        x1 = TruncNormalSampler(mu, sigma, pmin, pmax)
        # this should almost never fail
        assert -5 < (x1 - mu) / sigma < 5, msg2

        # vector argument
        x = np.array([-0.2, 0.1, 0.5])

        assert np.allclose(
            TruncNormalPrior(x, mu, sigma, pmin, pmax), scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        x1, x2 = (
            TruncNormalSampler(mu, sigma, pmin, pmax, size=10),
            scipy.stats.truncnorm.rvs(a, b, mu, sigma, size=10),
        )
        assert x1.shape == x2.shape, msg2

        # vector bounds; note the different semantics from `NormalPrior`,
        # which returns a vector consistently with `UniformPrior`
        mu, sigma = np.array([0.1, 0.15, 0.2]), np.array([2, 1, 2])
        pmin, pmax = np.array([-2, -2, -3]), np.array([1, 2, 1])
        a, b = (pmin - mu) / sigma, (pmax - mu) / sigma
        assert np.allclose(
            TruncNormalPrior(x, mu, sigma, pmin, pmax), scipy.stats.truncnorm.pdf(x, a, b, mu, sigma)
        ), msg1

        x1, x2 = TruncNormalSampler(mu, sigma, pmin, pmax), scipy.stats.truncnorm.rvs(a, b, mu, sigma)
        assert x1.shape == x2.shape, msg2

    def test_normalandtruncnormal(self):
        mu, sigma = 0, 1

        msg3 = "Normal and [-inf, inf] TruncNormal do not match"

        paramA = Normal(mu, sigma)("A")
        paramB = TruncNormal(mu, sigma, -np.inf, np.inf)("B")
        xs = np.linspace(-3, 3, 20)
        assert np.allclose(paramA.get_pdf(xs), paramB.get_pdf(xs)), msg3

    def test_metaparam(self):
        mu = Uniform(-1, 1)("mean")
        sigma = 2
        pmin, pmax = -3, 3

        msg4 = "problem with meta-parameter in TruncNormal"
        zeros = np.zeros(2)

        paramA = TruncNormal(mu, sigma, pmin, pmax)("A")
        xs = np.array([-3.5, 3.5])
        assert np.alltrue(paramA.get_pdf(xs, mu=mu.sample()) == zeros), msg4
