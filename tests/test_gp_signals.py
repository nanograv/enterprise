#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signals
----------------------------------

Tests for signal modules.
"""


import unittest
import numpy as np

from enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.signal_base import Function
from enterprise.signals.selections import Selection
import enterprise.signals.gp_signals as gs
from enterprise.signals import utils


class TestGPSignals(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psr = Pulsar(datadir + '/B1855+09_NANOGrav_11yv0.gls.par',
                          datadir + '/B1855+09_NANOGrav_11yv0.tim')

    def test_ecorr(self):
        """Test that ecorr signal returns correct values."""
        # set up signal parameter
        ecorr = parameter.Uniform(-10, -5)
        ec = gs.EcorrBasisModel(log10_ecorr=ecorr)
        ecm = ec(self.psr)

        # parameters
        ecorr = -6.4
        params = {'B1855+09_log10_ecorr': ecorr}

        # basis matrix test
        U = utils.create_quantization_matrix(self.psr.toas)
        msg = 'U matrix incorrect for Basis Ecorr signal.'
        assert np.allclose(U, ecm.get_basis(params)), msg

        # Jvec test
        jvec = 10**(2*ecorr) * np.ones(U.shape[1])
        msg = 'Prior vector incorrect for Basis Ecorr signal.'
        assert np.all(ecm.get_phi(params)==jvec), msg

        # inverse Jvec test
        msg = 'Prior vector inverse incorrect for Basis Ecorr signal.'
        assert np.all(ecm.get_phiinv(params)==1/jvec), msg

        # test shape
        msg = 'U matrix shape incorrect'
        assert ecm.basis_shape == U.shape, msg

    def test_fourier_red_noise(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pl = Function(utils.powerlaw, log10_A=parameter.Uniform(-18,-12),
                      gamma=parameter.Uniform(1,7))
        rn = gs.FourierBasisGP(spectrum=pl, components=30)
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        params = {'B1855+09_log10_A': log10_A,
                  'B1855+09_gamma': gamma}

        # basis matrix test
        F, f2, _ = utils.createfourierdesignmatrix_red(
            self.psr.toas, nmodes=30, freq=True)
        msg = 'F matrix incorrect for GP Fourier signal.'
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma) * f2[0]
        msg = 'Spectrum incorrect for GP Fourier signal.'
        assert np.all(rnm.get_phi(params)==phi), msg

        # inverse spectrum test
        msg = 'Spectrum inverse incorrect for GP Fourier signal.'
        assert np.all(rnm.get_phiinv(params)==1/phi), msg

        # test shape
        msg = 'F matrix shape incorrect'
        assert rnm.basis_shape == F.shape, msg

    def test_gp_timing_model(self):
        """Test that the timing model signal returns correct values."""
        # set up signal parameter
        ts = gs.TimingModel()
        tm = ts(self.psr)

        # basis matrix test
        msg = 'M matrix incorrect for Timing Model signal.'
        assert np.allclose(self.psr.Mmat, tm.get_basis()), msg

        # Jvec test
        phi = np.ones(self.psr.Mmat.shape[1]) * 1e40
        msg = 'Prior vector incorrect for Timing Model signal.'
        assert np.all(tm.get_phi()==phi), msg

        # inverse Jvec test
        msg = 'Prior vector inverse incorrect for Timing Model signal.'
        assert np.all(tm.get_phiinv()==1/phi), msg

        # test shape
        msg = 'M matrix shape incorrect'
        assert tm.basis_shape == self.psr.Mmat.shape, msg
