#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parameter
----------------------------------

Tests for ``parameter`` module.
"""


import unittest
from enterprise.signals.parameter import Parameter
from enterprise.signals import prior
import scipy.stats
import numpy as np


class TestPulsar(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Parameter class
        self.p = Parameter(name='x', value=2.4,
                           description='test parameter',
                           uncertainty=0.1,
                           prior=prior.Prior(prior.UniformUnnormedRV()))

    def test_prior(self):
        """Test parameter prior method."""

        msg1 = 'log-Prior not evaluated correctly'
        msg2 = 'Prior not evaluated correctly'
        assert self.p.prior_pdf() == 0, msg1
        assert self.p.prior_pdf(logpdf=False) == 1, msg2

    def test_set_prior(self):
        """Test setting prior."""

        # set new prior
        mean, std = 0, 1
        self.p.prior = prior.Prior(scipy.stats.norm(loc=mean, scale=std))

        m = 1.4
        msg = 'Gaussian prior set incorrectly'
        p1 = self.p.prior_pdf(m, logpdf=False)
        p2 = self.p.prior_pdf(m, logpdf=True)
        assert np.allclose(p1, 0.149727465636), msg
        assert np.allclose(p2, -1.8989385332), msg

    def test_wrong_prior_input(self):
        """Test error for wrong prior setting."""

        with self.assertRaises(ValueError) as context:
            self.p.prior = 5

        msg = 'ERROR: prior must be instance of Prior().'
        self.assertTrue(msg in context.exception)
