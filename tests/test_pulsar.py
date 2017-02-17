#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pulsar
----------------------------------

Tests for `pulsar` module. Will eventually want to add tests
for time slicing, PINT integration and pickling.
"""


import unittest
from enterprise_test_data import datadir
from enterprise.pulsar import Pulsar


class TestPulsar(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psr = Pulsar(datadir + '/B1855+09_NANOGrav_11yv0.gls.par',
                          datadir + '/B1855+09_NANOGrav_11yv0.tim')

    def test_residuals(self):
        """Check Residual shape."""

        msg = 'Residuals shape incorrect'
        assert self.psr.residuals.shape == (5634,), msg

    def test_toaerrs(self):
        """Check TOA errors shape."""

        msg = 'TOA errors shape incorrect'
        assert self.psr.toaerrs.shape == (5634,), msg

    def test_toas(self):
        """Check TOA shape."""

        msg = 'TOA shape incorrect'
        assert self.psr.toas.shape == (5634,), msg

    def test_freqs(self):
        """Check frequencies shape."""

        msg = 'Frequencies shape incorrect'
        assert self.psr.freqs.shape == (5634,), msg

    def test_sky(self):
        """Check Sky location."""

        sky = (1.4023094090612354, 4.9533700717180027)

        msg = 'Incorrect sky location'
        assert (self.psr.theta, self.psr.phi) == sky, msg

    def test_design_matrix(self):
        """Check design matrix shape."""

        msg = 'Design matrix shape incorrect.'
        assert self.psr.Mmat.shape == (5634, 120), msg

    def test_filter_data(self):
        """Place holder for filter_data tests."""
        assert self.psr.filter_data() is None

    def test_to_pickle(self):
        """Place holder for to_pickle tests."""
        assert self.psr.to_pickle('./') is None

    def test_wrong_input(self):
        """Test exception when incorrect par(tim) file given."""

        with self.assertRaises(IOError) as context:
            Pulsar('wrong.par', 'wrong.tim')

        msg = 'Cannot find parfile wrong.par or timfile wrong.tim!'
        self.assertTrue(msg in context.exception)
