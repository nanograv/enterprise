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

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, utils, white_signals
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


class Woodbury(object):
    def __init__(self, N, U, J):
        self.N = N
        self.U = U
        self.J = J

    def solve(self, other):
        if other.ndim == 1:
            Nx = np.array(other / self.N)
        elif other.ndim == 2:
            Nx = np.array(other / self.N[:, None])
        UNx = np.dot(self.U.T, Nx)

        Sigma = np.diag(1 / self.J) + np.dot(self.U.T, self.U / self.N[:, None])
        cf = sl.cho_factor(Sigma)
        if UNx.ndim == 1:
            tmp = np.dot(self.U, sl.cho_solve(cf, UNx)) / self.N
        else:
            tmp = np.dot(self.U, sl.cho_solve(cf, UNx)) / self.N[:, None]
        return Nx - tmp

    def logdet(self):
        Sigma = np.diag(1 / self.J) + np.dot(self.U.T, self.U / self.N[:, None])
        cf = sl.cho_factor(Sigma)
        ld = np.sum(np.log(self.N)) + np.sum(np.log(self.J))
        ld += np.sum(2 * np.log(np.diag(cf[0])))
        return ld


class TestWhiteSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

        # IPTA-like pulsar
        cls.ipsr = Pulsar(datadir + "/1713.Sep.T2.par", datadir + "/1713.Sep.T2.tim")

    def test_efac(self):
        """Test that efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        ef = white_signals.MeasurementNoise(efac=efac)
        efm = ef(self.psr)

        # parameters
        efac = 1.5
        params = {"B1855+09_efac": efac}

        # correct value
        nvec0 = efac ** 2 * self.psr.toaerrs ** 2

        # test
        msg = "EFAC covariance incorrect."
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_efac_backend(self):
        """Test that backend-efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        selection = Selection(selections.by_backend)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        efm = ef(self.psr)

        # parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        params = {
            "B1855+09_430_ASP_efac": efacs[0],
            "B1855+09_430_PUPPI_efac": efacs[1],
            "B1855+09_L-wide_ASP_efac": efacs[2],
            "B1855+09_L-wide_PUPPI_efac": efacs[3],
        }

        # correct value
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct] ** 2 * self.psr.toaerrs[ind] ** 2

        # test
        msg = "EFAC covariance incorrect."
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_equad(self):
        """Test that equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        eq = white_signals.EquadNoise(log10_equad=equad)
        eqm = eq(self.psr)

        # parameters
        equad = -6.4
        params = {"B1855+09_log10_equad": equad}

        # correct value
        nvec0 = 10 ** (2 * equad) * np.ones_like(self.psr.toas)

        # test
        msg = "EQUAD covariance incorrect."
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_equad_backend(self):
        """Test that backend-equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        selection = Selection(selections.by_backend)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
        eqm = eq(self.psr)

        # parameters
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {
            "B1855+09_430_ASP_log10_equad": equads[0],
            "B1855+09_430_PUPPI_log10_equad": equads[1],
            "B1855+09_L-wide_ASP_log10_equad": equads[2],
            "B1855+09_L-wide_PUPPI_log10_equad": equads[3],
        }

        # correct value
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = 10 ** (2 * equads[ct]) * np.ones(np.sum(ind))

        # test
        msg = "EQUAD covariance incorrect."
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad(self):
        """Test that addition of efac and equad signal returns
        correct covariance.
        """
        # set up signals
        efac = parameter.Uniform(0.1, 5)
        ef = white_signals.MeasurementNoise(efac=efac)
        equad = parameter.Uniform(-10, -5)
        eq = white_signals.EquadNoise(log10_equad=equad)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efac = 1.5
        equad = -6.4
        params = {"B1855+09_efac": efac, "B1855+09_log10_equad": equad}

        # correct value
        nvec0 = efac ** 2 * self.psr.toaerrs ** 2
        nvec0 += 10 ** (2 * equad) * np.ones_like(self.psr.toas)

        # test
        msg = "EFAC/EQUAD covariance incorrect."
        assert np.all(m.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad_backend(self):
        """Test that addition of efac-backend and equad-backend signal returns
        correct covariance.
        """
        selection = Selection(selections.by_backend)

        efac = parameter.Uniform(0.1, 5)
        equad = parameter.Uniform(-10, -5)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {
            "B1855+09_430_ASP_efac": efacs[0],
            "B1855+09_430_PUPPI_efac": efacs[1],
            "B1855+09_L-wide_ASP_efac": efacs[2],
            "B1855+09_L-wide_PUPPI_efac": efacs[3],
            "B1855+09_430_ASP_log10_equad": equads[0],
            "B1855+09_430_PUPPI_log10_equad": equads[1],
            "B1855+09_L-wide_ASP_log10_equad": equads[2],
            "B1855+09_L-wide_PUPPI_log10_equad": equads[3],
        }

        # correct value
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct] ** 2 * self.psr.toaerrs[ind] ** 2
            nvec0[ind] += 10 ** (2 * equads[ct]) * np.ones(np.sum(ind))

        logdet = np.sum(np.log(nvec0))

        # test
        msg = "EFAC/EQUAD covariance incorrect."
        assert np.all(m.get_ndiag(params) == nvec0), msg

        msg = "EFAC/EQUAD logdet incorrect."
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.psr.residuals, logdet=True)[1], logdet, rtol=1e-10), msg

        msg = "EFAC/EQUAD D1 solve incorrect."
        assert np.allclose(N.solve(self.psr.residuals), self.psr.residuals / nvec0, rtol=1e-10), msg

        msg = "EFAC/EQUAD 1D1 solve incorrect."
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=self.psr.residuals),
            np.dot(self.psr.residuals / nvec0, self.psr.residuals),
            rtol=1e-10,
        ), msg

        msg = "EFAC/EQUAD 2D1 solve incorrect."
        T = self.psr.Mmat
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=T), np.dot(T.T, self.psr.residuals / nvec0), rtol=1e-10
        ), msg

        msg = "EFAC/EQUAD 2D2 solve incorrect."
        assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, T / nvec0[:, None]), rtol=1e-10), msg

    def _ecorr_test(self, method="sparse"):
        """Test of sparse/sherman-morrison ecorr signal and solve methods."""
        selection = Selection(selections.by_backend)

        efac = parameter.Uniform(0.1, 5)
        ecorr = parameter.Uniform(-10, -5)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection, method=method)
        tm = gp_signals.TimingModel()
        s = ef + ec + tm
        m = s(self.psr)

        # set parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        ecorrs = [-6.1, -6.2, -6.3, -6.4]
        params = {
            "B1855+09_430_ASP_efac": efacs[0],
            "B1855+09_430_PUPPI_efac": efacs[1],
            "B1855+09_L-wide_ASP_efac": efacs[2],
            "B1855+09_L-wide_PUPPI_efac": efacs[3],
            "B1855+09_430_ASP_log10_ecorr": ecorrs[0],
            "B1855+09_430_PUPPI_log10_ecorr": ecorrs[1],
            "B1855+09_L-wide_ASP_log10_ecorr": ecorrs[2],
            "B1855+09_L-wide_PUPPI_log10_ecorr": ecorrs[3],
        }

        # get EFAC Nvec
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct] ** 2 * self.psr.toaerrs[ind] ** 2

        # get the basis
        bflags = self.psr.backend_flags
        Umats = []
        for flag in np.unique(bflags):
            mask = bflags == flag
            Umats.append(utils.create_quantization_matrix(self.psr.toas[mask], nmin=2)[0])
        nepoch = sum(U.shape[1] for U in Umats)
        U = np.zeros((len(self.psr.toas), nepoch))
        jvec = np.zeros(nepoch)
        netot = 0
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            nn = Umats[ct].shape[1]
            U[mask, netot : nn + netot] = Umats[ct]
            jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ct])
            netot += nn

        # get covariance matrix
        wd = Woodbury(nvec0, U, jvec)

        # test
        msg = "EFAC/ECORR {} logdet incorrect.".format(method)
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.psr.residuals, logdet=True)[1], wd.logdet(), rtol=1e-10), msg

        msg = "EFAC/ECORR {} D1 solve incorrect.".format(method)
        assert np.allclose(N.solve(self.psr.residuals), wd.solve(self.psr.residuals), rtol=1e-10), msg

        msg = "EFAC/ECORR {} 1D1 solve incorrect.".format(method)
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=self.psr.residuals),
            np.dot(self.psr.residuals, wd.solve(self.psr.residuals)),
            rtol=1e-10,
        ), msg

        msg = "EFAC/ECORR {} 2D1 solve incorrect.".format(method)
        T = m.get_basis()
        assert np.allclose(
            N.solve(self.psr.residuals, left_array=T), np.dot(T.T, wd.solve(self.psr.residuals)), rtol=1e-10
        ), msg

        msg = "EFAC/ECORR {} 2D2 solve incorrect.".format(method)
        assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, wd.solve(T)), rtol=1e-10), msg

    def _ecorr_test_ipta(self, method="sparse"):
        """Test of sparse/sherman-morrison ecorr signal and solve methods."""
        selection = Selection(selections.nanograv_backends)

        efac = parameter.Uniform(0.1, 5)
        ecorr = parameter.Uniform(-10, -5)
        ef = white_signals.MeasurementNoise(efac=efac)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection, method=method)
        tm = gp_signals.TimingModel()
        s = ef + ec + tm
        m = s(self.ipsr)

        # set parameters
        efacs = [1.3]
        ecorrs = [-6.1, -6.2, -6.3, -6.4, -7.2, -8.4, -7.1, -7.9]
        params = {
            "J1713+0747_efac": efacs[0],
            "J1713+0747_ASP-L_log10_ecorr": ecorrs[0],
            "J1713+0747_ASP-S_log10_ecorr": ecorrs[1],
            "J1713+0747_GASP-8_log10_ecorr": ecorrs[2],
            "J1713+0747_GASP-L_log10_ecorr": ecorrs[3],
            "J1713+0747_GUPPI-8_log10_ecorr": ecorrs[4],
            "J1713+0747_GUPPI-L_log10_ecorr": ecorrs[5],
            "J1713+0747_PUPPI-L_log10_ecorr": ecorrs[6],
            "J1713+0747_PUPPI-S_log10_ecorr": ecorrs[7],
        }

        # get EFAC Nvec
        nvec0 = efacs[0] ** 2 * self.ipsr.toaerrs ** 2

        # get the basis
        flags = ["ASP-L", "ASP-S", "GASP-8", "GASP-L", "GUPPI-8", "GUPPI-L", "PUPPI-L", "PUPPI-S"]
        bflags = self.ipsr.backend_flags
        Umats = []
        for flag in np.unique(bflags):
            if flag in flags:
                mask = bflags == flag
                Umats.append(utils.create_quantization_matrix(self.ipsr.toas[mask], nmin=2)[0])
        nepoch = sum(U.shape[1] for U in Umats)
        U = np.zeros((len(self.ipsr.toas), nepoch))
        jvec = np.zeros(nepoch)
        netot, ct = 0, 0
        for flag in np.unique(bflags):
            if flag in flags:
                mask = bflags == flag
                nn = Umats[ct].shape[1]
                U[mask, netot : nn + netot] = Umats[ct]
                jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ct])
                netot += nn
                ct += 1

        # get covariance matrix
        wd = Woodbury(nvec0, U, jvec)

        # test
        msg = "EFAC/ECORR {} logdet incorrect.".format(method)
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.ipsr.residuals, logdet=True)[1], wd.logdet(), rtol=1e-8), msg

        msg = "EFAC/ECORR {} D1 solve incorrect.".format(method)
        assert np.allclose(N.solve(self.ipsr.residuals), wd.solve(self.ipsr.residuals), rtol=1e-8), msg

        msg = "EFAC/ECORR {} 1D1 solve incorrect.".format(method)
        assert np.allclose(
            N.solve(self.ipsr.residuals, left_array=self.ipsr.residuals),
            np.dot(self.ipsr.residuals, wd.solve(self.ipsr.residuals)),
            rtol=1e-8,
        ), msg

        msg = "EFAC/ECORR {} 2D1 solve incorrect.".format(method)
        T = m.get_basis()
        assert np.allclose(
            N.solve(self.ipsr.residuals, left_array=T), np.dot(T.T, wd.solve(self.ipsr.residuals)), rtol=1e-8
        ), msg

        msg = "EFAC/ECORR {} 2D2 solve incorrect.".format(method)
        assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, wd.solve(T)), rtol=1e-8), msg

    def test_ecorr_sparse(self):
        """Test of sparse ecorr signal and solve methods."""
        self._ecorr_test(method="sparse")

    def test_ecorr_sherman_morrison(self):
        """Test of sherman-morrison ecorr signal and solve methods."""
        self._ecorr_test(method="sherman-morrison")

    def test_ecorr_block(self):
        """Test of block matrix ecorr signal and solve methods."""
        self._ecorr_test(method="block")

    def test_ecorr_sparse_ipta(self):
        """Test of sparse ecorr signal and solve methods."""
        self._ecorr_test_ipta(method="sparse")

    def test_ecorr_sherman_morrison_ipta(self):
        """Test of sherman-morrison ecorr signal and solve methods."""
        self._ecorr_test_ipta(method="sherman-morrison")

    def test_ecorr_block_ipta(self):
        """Test of block matrix ecorr signal and solve methods."""
        self._ecorr_test_ipta(method="block")


class TestWhiteSignalsPint(TestWhiteSignals):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            timing_package="pint",
        )

        # IPTA-like pulsar
        cls.ipsr = Pulsar(
            datadir + "/1713.Sep.T2.par", datadir + "/1713.Sep.T2.tim", ephem="DE421", timint_package="pint"
        )
