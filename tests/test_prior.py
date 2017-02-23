#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_prior
----------------------------------

Tests for `prior` module.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import unittest

from enterprise.signals.prior import Prior
from enterprise.signals.prior import (UniformUnnormedRV, UniformBoundedRV,
                                      GaussianBoundedRV)
from scipy.stats import truncnorm


class TestPrior(unittest.TestCase):

    def setUp(self):
        """Setup the Prior object."""
        # A half-bounded uniform prior ensuring parm >= 0.
        self.uPrior = Prior(UniformUnnormedRV(lower=0.))

        # A normalized uniform prior ensuring param [10^-18, 10^-12]
        self.bPrior = Prior(UniformBoundedRV(1.0e-18, 1.0e-12))

        # A bounded Gaussian prior to ensure that param is in [0, 1]
        mean, std, low, up = 0.9, 0.1, 0.0, 1.0
        self.gPrior = Prior(GaussianBoundedRV(loc=mean, scale=std,
                                              lower=low, upper=up))

    def test_unnormed_uniform_prior(self):
        """check UniformUnnormedRV"""
        msg = "UniformUnnormed prior: incorrect test {0}"
        test_vals = [-0.5, 0.5, 1.0]
        correct = [0.0, 1.0, 1.0]  # correct
        for ii, xx in enumerate(test_vals):
            assert self.uPrior.pdf(xx) == correct[ii], msg.format(ii)

    def test_uniform_bounded_prior(self):
        """check UniformBoundedRV"""
        msg = "UniformBounded prior: incorrect test {0}"
        test_vals = [-0.5, 0.0, 1.0e-15, 1.0]
        correct = [0.0, 0.0, 1.0 / (1.0e-12 - 1.0e-18), 0.0]  # correct
        for ii, xx in enumerate(test_vals):
            assert self.bPrior.pdf(xx) == correct[ii], msg.format(ii)

    def test_truncnorm_prior(self):
        """check truncnorm RV"""
        msg = "truncnorm prior: incorrect test {0}"
        test_vals = [-0.1, 0.0, 0.5, 1.0, 1.1]
        mean, std, low, up = 0.9, 0.1, 0.0, 1.0
        a, b = (low - mean) / std, (up - mean) / std
        correct = truncnorm(loc=mean, scale=std, a=a, b=b)
        for ii, xx in enumerate(test_vals):
            assert self.gPrior.pdf(xx) == correct.pdf(xx), msg.format(ii)
