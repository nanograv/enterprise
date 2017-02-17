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
