#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `utils` module.
"""

import unittest
import numpy as np
from enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import utils
import enterprise.constants as const


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

    def test_ecc_cw_waveform(self):
        """Check eccentric wafeform generation."""
        nmax = 100
        mc = 5e8
        dl = 300
        h0 = 1e-14
        F = 2e-8
        e = 0.6
        t = self.psr.toas
        l0 = 0.2
        gamma = 0.4
        gammadot = 0.1
        inc = 1.3
        s = utils.calculate_splus_scross(nmax, mc, dl, h0, F, e,
                                         t, l0, gamma, gammadot, inc)

        msg = 'Single source waveform shape incorrect'
        assert s[0].shape == (5634,), msg
        assert s[1].shape == (5634,), msg

    def test_fplus_fcross(self):
        """Check fplus, fcross generation."""
        gwtheta = 1.4
        gwphi = 2.7
        fplus, fcross = utils.fplus_fcross(self.psr.theta, self.psr.phi,
                                           gwtheta, gwphi)

        msg1 = 'Fplus value incorrect'
        msg2 = 'Fcross value incorrect'
        assert np.allclose(fplus, 0.161508137208), msg1
        assert np.allclose(fcross, -0.130823200124), msg2

    def test_numerical_ecc_integration(self):
        """Test numerical integration of eccentric GW."""
        F0 = 1e-8
        e0 = 0.3
        gamma0 = 0.4
        phase0 = 1.2
        mc = 1e9
        q = 0.25
        t = self.psr.toas - self.psr.toas.min()
        ind = np.argsort(t)
        s = utils.solve_coupled_ecc_solution(F0, e0, gamma0,
                                             phase0, mc, q, t[ind])
        s2 = utils.solve_coupled_constecc_solution(F0, e0, phase0, mc, t[ind])
        msg = 'Numerical integration failed'
        assert s.shape == (5634, 4), msg
        assert s2.shape == (5634, 2), msg

    def test_quantization_matrix(self):
        """Test quantization matrix generation."""
        U = utils.create_quantization_matrix(self.psr.toas, dt=1)

        msg1 = 'Quantization matrix shape incorrect.'
        msg2 = 'Quantization matrix contains single TOA epochs.'
        assert U.shape == (5634, 294), msg1
        assert all(np.sum(U, axis=0) > 1), msg2

    def test_psd(self):
        """Test PSD functions."""
        Tmax = self.psr.toas.max() - self.psr.toas.min()
        f = np.linspace(1/Tmax, 10/Tmax, 10)
        log10_A = -15
        gamma = 4.33
        lf0 = -8.5
        kappa = 10/3
        beta = 0.5
        pl = ((10**log10_A)**2 / 12.0 / np.pi**2 *
              const.fyr**(gamma-3) * f**(-gamma))
        hcf = (10**log10_A * (f / const.fyr) ** ((3-gamma) / 2) /
               (1 + (10**lf0 / f) ** kappa) ** beta)
        pt = hcf**2/12/np.pi**2/f**3

        msg = 'PSD calculation incorrect'
        assert np.all(utils.powerlaw(f, log10_A, gamma) == pl), msg
        assert np.all(utils.turnover(f, log10_A, gamma,
                                     lf0, kappa, beta) == pt), msg
