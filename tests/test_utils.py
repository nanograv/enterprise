#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `utils` module.
"""

import unittest
from enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psr = Pulsar(datadir + '/B1855+09_NANOGrav_11yv0.gls.par',
                          datadir + '/B1855+09_NANOGrav_11yv0.tim')
        self.F, _ = utils.createfourierdesignmatrix(t=self.psr.toas, nmodes=30)

    def test_createfourierdesignmatrix(self, nf=30):
        """Check Fourier design matrix shape."""

        msg = 'Residuals shape incorrect'
        assert self.F.shape == (5634, 2 * nf), msg
