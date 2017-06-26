#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_white_signals
----------------------------------

Tests for white signal modules.
"""


import unittest
import numpy as np
import scipy.linalg as sl

from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
import enterprise.signals.white_signals as ws
import enterprise.signals.gp_signals as gp
from enterprise.signals import utils


class TestWhiteSignals(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                          datadir + '/B1855+09_NANOGrav_9yv1.tim')

    def test_efac(self):
        """Test that efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        ef = ws.MeasurementNoise(efac=efac)
        efm = ef(self.psr)

        # parameters
        efac = 1.5
        params = {'B1855+09_efac': efac}

        # correct value
        nvec0 = efac**2 * self.psr.toaerrs**2

        # test
        msg = 'EFAC covariance incorrect.'
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_efac_backend(self):
        """Test that backend-efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        selection = Selection(selections.by_backend)
        ef = ws.MeasurementNoise(efac=efac, selection=selection)
        efm = ef(self.psr)

        # parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        params = {'B1855+09_efac_430_ASP': efacs[0],
                  'B1855+09_efac_430_PUPPI': efacs[1],
                  'B1855+09_efac_L-wide_ASP': efacs[2],
                  'B1855+09_efac_L-wide_PUPPI': efacs[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct]**2 * self.psr.toaerrs[ind]**2

        # test
        msg = 'EFAC covariance incorrect.'
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_equad(self):
        """Test that equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        eq = ws.EquadNoise(log10_equad=equad)
        eqm = eq(self.psr)

        # parameters
        equad = -6.4
        params = {'B1855+09_log10_equad': equad}

        # correct value
        nvec0 = 10**(2*equad) * np.ones_like(self.psr.toas)

        # test
        msg = 'EQUAD covariance incorrect.'
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_equad_backend(self):
        """Test that backend-equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        selection = Selection(selections.by_backend)
        eq = ws.EquadNoise(log10_equad=equad, selection=selection)
        eqm = eq(self.psr)

        # parameters
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_log10_equad_430_ASP': equads[0],
                  'B1855+09_log10_equad_430_PUPPI': equads[1],
                  'B1855+09_log10_equad_L-wide_ASP': equads[2],
                  'B1855+09_log10_equad_L-wide_PUPPI': equads[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = 10**(2*equads[ct]) * np.ones(np.sum(ind))

        # test
        msg = 'EQUAD covariance incorrect.'
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad(self):
        """Test that addition of efac and equad signal returns
        correct covariance.
        """
        # set up signals
        efac = parameter.Uniform(0.1, 5)
        ef = ws.MeasurementNoise(efac=efac)
        equad = parameter.Uniform(-10, -5)
        eq = ws.EquadNoise(log10_equad=equad)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efac = 1.5
        equad = -6.4
        params = {'B1855+09_efac': efac,
                  'B1855+09_log10_equad': equad}

        # correct value
        nvec0 = efac**2 * self.psr.toaerrs**2
        nvec0 += 10**(2*equad) * np.ones_like(self.psr.toas)

        # test
        msg = 'EFAC/EQUAD covariance incorrect.'
        assert np.all(m.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad_backend(self):
        """Test that addition of efac-backend and equad-backend signal returns
        correct covariance.
        """
        selection = Selection(selections.by_backend)

        efac = parameter.Uniform(0.1, 5)
        equad = parameter.Uniform(-10, -5)
        ef = ws.MeasurementNoise(efac=efac, selection=selection)
        eq = ws.EquadNoise(log10_equad=equad, selection=selection)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_efac_430_ASP': efacs[0],
                  'B1855+09_efac_430_PUPPI': efacs[1],
                  'B1855+09_efac_L-wide_ASP': efacs[2],
                  'B1855+09_efac_L-wide_PUPPI': efacs[3],
                  'B1855+09_log10_equad_430_ASP': equads[0],
                  'B1855+09_log10_equad_430_PUPPI': equads[1],
                  'B1855+09_log10_equad_L-wide_ASP': equads[2],
                  'B1855+09_log10_equad_L-wide_PUPPI': equads[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct]**2 * self.psr.toaerrs[ind]**2
            nvec0[ind] += 10**(2*equads[ct]) * np.ones(np.sum(ind))

        logdet = np.sum(np.log(nvec0))

        # test
        msg = 'EFAC/EQUAD covariance incorrect.'
        assert np.all(m.get_ndiag(params) == nvec0), msg

        msg = 'EFAC/EQUAD logdet incorrect.'
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.psr.residuals, logdet=True)[1],
                           logdet, rtol=1e-10), msg

        msg = 'EFAC/EQUAD D1 solve incorrect.'
        assert np.allclose(N.solve(self.psr.residuals),
                           self.psr.residuals/nvec0,
                           rtol=1e-10), msg

        msg = 'EFAC/EQUAD 1D1 solve incorrect.'
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=self.psr.residuals),
            np.dot(self.psr.residuals/nvec0, self.psr.residuals),
            rtol=1e-10), msg

        msg = 'EFAC/EQUAD 2D1 solve incorrect.'
        T = self.psr.Mmat
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=T),
            np.dot(T.T, self.psr.residuals/nvec0),
            rtol=1e-10), msg

        msg = 'EFAC/EQUAD 2D2 solve incorrect.'
        assert np.allclose(
            N.solve(T, left_array=T),
            np.dot(T.T, T/nvec0[:,None]),
            rtol=1e-10), msg

    def _ecorr_test(self, method='sparse'):
        """Test of sparse/sherman-morrison ecorr signal and solve methods."""
        selection = Selection(selections.by_backend)

        efac = parameter.Uniform(0.1, 5)
        ecorr = parameter.Uniform(-10, -5)
        ef = ws.MeasurementNoise(efac=efac, selection=selection)
        ec = ws.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection,
                                 method=method)
        tm = gp.TimingModel()
        s = ef + ec + tm
        m = s(self.psr)

        # set parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        ecorrs = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_efac_430_ASP': efacs[0],
                  'B1855+09_efac_430_PUPPI': efacs[1],
                  'B1855+09_efac_L-wide_ASP': efacs[2],
                  'B1855+09_efac_L-wide_PUPPI': efacs[3],
                  'B1855+09_log10_ecorr_430_ASP': ecorrs[0],
                  'B1855+09_log10_ecorr_430_PUPPI': ecorrs[1],
                  'B1855+09_log10_ecorr_L-wide_ASP': ecorrs[2],
                  'B1855+09_log10_ecorr_L-wide_PUPPI': ecorrs[3]}

        # get EFAC Nvec
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct]**2 * self.psr.toaerrs[ind]**2

        # get the basis
        bflags = self.psr.backend_flags
        Umats = []
        for flag in np.unique(bflags):
            mask = bflags == flag
            Umats.append(utils.create_quantization_matrix(
                self.psr.toas[mask], nmin=2)[0])
        nepoch = sum(U.shape[1] for U in Umats)
        U = np.zeros((len(self.psr.toas), nepoch))
        jvec = np.zeros(nepoch)
        netot = 0
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            nn = Umats[ct].shape[1]
            U[mask, netot:nn+netot] = Umats[ct]
            jvec[netot:nn+netot] = 10**(2*ecorrs[ct])
            netot += nn

        # get covariance matrix
        cov = np.diag(nvec0) + np.dot(U*jvec[None, :], U.T)
        cf = sl.cho_factor(cov)
        logdet = np.sum(2*np.log(np.diag(cf[0])))

        # test
        msg = 'EFAC/ECORR {} logdet incorrect.'.format(method)
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.psr.residuals, logdet=True)[1],
                           logdet, rtol=1e-10), msg

        msg = 'EFAC/ECORR {} D1 solve incorrect.'.format(method)
        assert np.allclose(N.solve(self.psr.residuals),
                           sl.cho_solve(cf, self.psr.residuals),
                           rtol=1e-10), msg

        msg = 'EFAC/ECORR {} 1D1 solve incorrect.'.format(method)
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=self.psr.residuals),
            np.dot(self.psr.residuals, sl.cho_solve(cf, self.psr.residuals)),
            rtol=1e-10), msg

        msg = 'EFAC/ECORR {} 2D1 solve incorrect.'.format(method)
        T = m.get_basis()
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=T),
            np.dot(T.T, sl.cho_solve(cf, self.psr.residuals)),
            rtol=1e-10), msg

        msg = 'EFAC/ECORR {} 2D2 solve incorrect.'.format(method)
        assert np.allclose(
            N.solve(T, left_array=T),
            np.dot(T.T, sl.cho_solve(cf, T)),
            rtol=1e-10), msg

    def test_ecorr_sparse(self):
        """Test of sparse ecorr signal and solve methods."""
        self._ecorr_test(method='sparse')

    def test_ecorr_sherman_morrison(self):
        """Test of sherman-morrison ecorr signal and solve methods."""
        self._ecorr_test(method='sherman-morrison')

    def test_ecorr_block(self):
        """Test of block matrix ecorr signal and solve methods."""
        self._ecorr_test(method='block')
