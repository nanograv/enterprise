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
        self.F, _ = utils.createfourierdesignmatrix_red(t=self.psr.toas,
                                                        nmodes=30)
        self.Fdm = utils.createfourierdesignmatrix_dm(t=self.psr.toas,
                                                      ssbfreqs=self.psr.freqs,
                                                      nmodes=30)
        tmp = utils.createfourierdesignmatrix_eph(t=self.psr.toas,
                                                  phi=self.psr.phi,
                                                  theta=self.psr.theta,
                                                  nmodes=30)
        self.Fx, self.Fy, self.Fz = tmp

        self.Mm = utils.create_stabletimingdesignmatrix(self.psr.Mmat)

    def test_createstabletimingdesignmatrix(self):
        """Timing model design matrix shape."""

        msg = 'Timing model design matrix shape incorrect'
        assert self.Mm.shape == self.psr.Mmat.shape, msg

    def test_createfourierdesignmatrix_red(self, nf=30):
        """Check Fourier design matrix shape."""

        msg = 'Fourier design matrix shape incorrect'
        assert self.F.shape == (5634, 2 * nf), msg

    def test_createfourierdesignmatrix_dm(self, nf=30):
        """Check DM-variation Fourier design matrix shape."""

        msg = 'DM-variation Fourier design matrix shape incorrect'
        assert self.Fdm.shape == (5634, 2 * nf), msg

    def test_createfourierdesignmatrix_ephx(self, nf=30):
        """Check x-axis ephemeris Fourier design matrix shape."""

        msg = 'Ephemeris x-axis Fourier design matrix shape incorrect'
        assert self.Fx.shape == (5634, 2 * nf), msg

    def test_createfourierdesignmatrix_ephy(self, nf=30):
        """Check y-axis ephemeris Fourier design matrix shape."""

        msg = 'Ephemeris y-axis Fourier design matrix shape incorrect'
        assert self.Fy.shape == (5634, 2 * nf), msg

    def test_createfourierdesignmatrix_ephz(self, nf=30):
        """Check z-axis ephemeris Fourier design matrix shape."""

        msg = 'Ephemeris z-axis Fourier design matrix shape incorrect'
        assert self.Fz.shape == (5634, 2 * nf), msg
