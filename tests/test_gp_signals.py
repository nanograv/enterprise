#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_gp_signals
----------------------------------

Tests for GP signal modules.
"""


import unittest

import numpy as np
import scipy.linalg as sl

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, signal_base, utils
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


@signal_base.function
def create_quant_matrix(toas, dt=1):
    U, _ = utils.create_quantization_matrix(toas, dt=dt, nmin=1)
    avetoas = np.array([toas[idx.astype(bool)].mean() for idx in U.T])
    # return value slightly different than 1 to get around ECORR columns
    return U * 1.0000001, avetoas


@signal_base.function
def se_kernel(etoas, log10_sigma=-7, log10_lam=np.log10(30 * 86400)):
    tm = np.abs(etoas[None, :] - etoas[:, None])
    d = np.eye(tm.shape[0]) * 10 ** (2 * (log10_sigma - 1.5))
    return 10 ** (2 * log10_sigma) * np.exp(-(tm ** 2) / 2 / 10 ** (2 * log10_lam)) + d


class TestGPSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_ecorr(self):
        """Test that ecorr signal returns correct values."""
        # set up signal parameter
        ecorr = parameter.Uniform(-10, -5)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr)
        ecm = ec(self.psr)

        # parameters
        ecorr = -6.4
        params = {"B1855+09_basis_ecorr_log10_ecorr": ecorr}

        # basis matrix test
        U = utils.create_quantization_matrix(self.psr.toas)[0]
        msg = "U matrix incorrect for Basis Ecorr signal."
        assert np.allclose(U, ecm.get_basis(params)), msg

        # Jvec test
        jvec = 10 ** (2 * ecorr) * np.ones(U.shape[1])
        msg = "Prior vector incorrect for Basis Ecorr signal."
        assert np.all(ecm.get_phi(params) == jvec), msg

        # inverse Jvec test
        msg = "Prior vector inverse incorrect for Basis Ecorr signal."
        assert np.all(ecm.get_phiinv(params) == 1 / jvec), msg

        # test shape
        msg = "U matrix shape incorrect"
        assert ecm.get_basis(params).shape == U.shape, msg

    def test_ecorr_backend(self):
        """Test that ecorr-backend signal returns correct values."""
        # set up signal parameter
        ecorr = parameter.Uniform(-10, -5)
        selection = Selection(selections.by_backend)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection)
        ecm = ec(self.psr)

        # parameters
        ecorrs = [-6.1, -6.2, -6.3, -6.4]
        params = {
            "B1855+09_basis_ecorr_430_ASP_log10_ecorr": ecorrs[0],
            "B1855+09_basis_ecorr_430_PUPPI_log10_ecorr": ecorrs[1],
            "B1855+09_basis_ecorr_L-wide_ASP_log10_ecorr": ecorrs[2],
            "B1855+09_basis_ecorr_L-wide_PUPPI_log10_ecorr": ecorrs[3],
        }

        # get the basis
        bflags = self.psr.backend_flags
        Umats = []
        for flag in np.unique(bflags):
            mask = bflags == flag
            Umats.append(utils.create_quantization_matrix(self.psr.toas[mask])[0])
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

        # basis matrix test
        msg = "U matrix incorrect for Basis Ecorr-backend signal."
        assert np.allclose(U, ecm.get_basis(params)), msg

        # Jvec test
        msg = "Prior vector incorrect for Basis Ecorr backend signal."
        assert np.all(ecm.get_phi(params) == jvec), msg

        # inverse Jvec test
        msg = "Prior vector inverse incorrect for Basis Ecorr backend signal."
        assert np.all(ecm.get_phiinv(params) == 1 / jvec), msg

        # test shape
        msg = "U matrix shape incorrect"
        assert ecm.get_basis(params).shape == U.shape, msg

    def test_kernel(self):

        log10_sigma = parameter.Uniform(-10, -5)
        log10_lam = parameter.Uniform(np.log10(86400), np.log10(1500 * 86400))
        basis = create_quant_matrix(dt=7 * 86400)
        prior = se_kernel(log10_sigma=log10_sigma, log10_lam=log10_lam)
        se = gp_signals.BasisGP(prior, basis, name="se")

        sem = se(self.psr)

        # parameters
        log10_lam, log10_sigma = 7.4, -6.4
        params = {"B1855+09_se_log10_lam": log10_lam, "B1855+09_se_log10_sigma": log10_sigma}

        # basis check
        U, avetoas = create_quant_matrix(self.psr.toas, dt=7 * 86400)
        msg = "Kernel Basis incorrect"
        assert np.allclose(U, sem.get_basis(params)), msg

        # kernel test
        K = se_kernel(avetoas, log10_lam=log10_lam, log10_sigma=log10_sigma)
        msg = "Kernel incorrect"
        assert np.allclose(K, sem.get_phi(params)), msg

        # inverse kernel test
        Kinv = np.linalg.inv(K)
        msg = "Kernel inverse incorrect"
        assert np.allclose(Kinv, sem.get_phiinv(params)), msg

    def test_kernel_backend(self):
        # set up signal parameter
        selection = Selection(selections.by_backend)
        log10_sigma = parameter.Uniform(-10, -5)
        log10_lam = parameter.Uniform(np.log10(86400), np.log10(1500 * 86400))
        basis = create_quant_matrix(dt=7 * 86400)
        prior = se_kernel(log10_sigma=log10_sigma, log10_lam=log10_lam)

        se = gp_signals.BasisGP(prior, basis, selection=selection, name="se")
        sem = se(self.psr)

        # parameters
        log10_sigmas = [-7, -6, -6.4, -8.5]
        log10_lams = [8.3, 7.4, 6.8, 5.6]
        params = {
            "B1855+09_se_430_ASP_log10_lam": log10_lams[0],
            "B1855+09_se_430_ASP_log10_sigma": log10_sigmas[0],
            "B1855+09_se_430_PUPPI_log10_lam": log10_lams[1],
            "B1855+09_se_430_PUPPI_log10_sigma": log10_sigmas[1],
            "B1855+09_se_L-wide_ASP_log10_lam": log10_lams[2],
            "B1855+09_se_L-wide_ASP_log10_sigma": log10_sigmas[2],
            "B1855+09_se_L-wide_PUPPI_log10_lam": log10_lams[3],
            "B1855+09_se_L-wide_PUPPI_log10_sigma": log10_sigmas[3],
        }

        # get the basis
        bflags = self.psr.backend_flags
        Fmats, fs, phis = [], [], []
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            U, avetoas = create_quant_matrix(self.psr.toas[mask], dt=7 * 86400)
            Fmats.append(U)
            fs.append(avetoas)
            phis.append(se_kernel(avetoas, log10_sigma=log10_sigmas[ct], log10_lam=log10_lams[ct]))

        nf = sum(F.shape[1] for F in Fmats)
        U = np.zeros((len(self.psr.toas), nf))
        K = sl.block_diag(*phis)
        Kinv = np.linalg.inv(K)
        nftot = 0
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            nn = Fmats[ct].shape[1]
            U[mask, nftot : nn + nftot] = Fmats[ct]
            nftot += nn

        msg = "Kernel basis incorrect for backend signal."
        assert np.allclose(U, sem.get_basis(params)), msg

        # spectrum test
        msg = "Kernel incorrect for backend signal."
        assert np.allclose(sem.get_phi(params), K), msg

        # inverse spectrum test
        msg = "Kernel inverse incorrect for backend signal."
        assert np.allclose(sem.get_phiinv(params), Kinv), msg

    def test_fourier_red_noise(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        params = {"B1855+09_red_noise_log10_A": log10_A, "B1855+09_red_noise_gamma": gamma}

        # basis matrix test
        F, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for GP Fourier signal."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma)
        msg = "Spectrum incorrect for GP Fourier signal."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier signal."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_fourier_red_noise_pshift(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, pshift=True, pseed=42)
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        params = {"B1855+09_red_noise_log10_A": log10_A, "B1855+09_red_noise_gamma": gamma}

        # basis matrix test
        F, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=30, pshift=True, pseed=42)
        msg = "F matrix incorrect for GP Fourier signal."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma)
        msg = "Spectrum incorrect for GP Fourier signal."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier signal."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_fourier_red_user_freq_array(self):
        """Test that red noise signal returns correct values with user defined
        frequency array."""
        # set parameters
        log10_A, gamma = -14.5, 4.33
        params = {"B1855+09_red_noise_log10_A": log10_A, "B1855+09_red_noise_gamma": gamma}

        F, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)

        # set up signal model. use list of frequencies to make basis
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, modes=f2[::2])
        rnm = rn(self.psr)

        # basis matrix test
        msg = "F matrix incorrect for GP Fourier signal."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma)
        msg = "Spectrum incorrect for GP Fourier signal."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier signal."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_fourier_red_noise_backend(self):
        """Test that red noise-backend signal returns correct values."""
        # set up signal parameter
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        selection = Selection(selections.by_backend)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, selection=selection)
        rnm = rn(self.psr)

        # parameters
        log10_As = [-14, -14.4, -15, -14.8]
        gammas = [2.3, 4.4, 1.8, 5.6]
        params = {
            "B1855+09_red_noise_430_ASP_gamma": gammas[0],
            "B1855+09_red_noise_430_PUPPI_gamma": gammas[1],
            "B1855+09_red_noise_L-wide_ASP_gamma": gammas[2],
            "B1855+09_red_noise_L-wide_PUPPI_gamma": gammas[3],
            "B1855+09_red_noise_430_ASP_log10_A": log10_As[0],
            "B1855+09_red_noise_430_PUPPI_log10_A": log10_As[1],
            "B1855+09_red_noise_L-wide_ASP_log10_A": log10_As[2],
            "B1855+09_red_noise_L-wide_PUPPI_log10_A": log10_As[3],
        }

        # get the basis
        bflags = self.psr.backend_flags
        Fmats, fs, phis = [], [], []
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            F, f = utils.createfourierdesignmatrix_red(self.psr.toas[mask], 30)
            Fmats.append(F)
            fs.append(f)
            phis.append(utils.powerlaw(f, log10_As[ct], gammas[ct]))

        nf = sum(F.shape[1] for F in Fmats)
        F = np.zeros((len(self.psr.toas), nf))
        phi = np.hstack([p for p in phis])
        nftot = 0
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            nn = Fmats[ct].shape[1]
            F[mask, nftot : nn + nftot] = Fmats[ct]
            nftot += nn

        msg = "F matrix incorrect for GP Fourier backend signal."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        msg = "Spectrum incorrect for GP Fourier backend signal."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier backend signal."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_red_noise_add(self):
        """Test that red noise addition only returns independent columns."""
        # set up signals
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        cpl = utils.powerlaw(
            log10_A=parameter.Uniform(-18, -12)("log10_Agw"), gamma=parameter.Uniform(1, 7)("gamma_gw")
        )

        # parameters
        log10_A, gamma = -14.5, 4.33
        log10_Ac, gammac = -15.5, 1.33
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "log10_Agw": log10_Ac,
            "gamma_gw": gammac,
        }

        Tmax = self.psr.toas.max() - self.psr.toas.min()
        tpars = [
            (30, 20, Tmax, Tmax),
            (20, 30, Tmax, Tmax),
            (30, 30, Tmax, Tmax),
            (30, 20, Tmax, 1.123 * Tmax),
            (20, 30, Tmax, 1.123 * Tmax),
            (30, 30, 1.123 * Tmax, Tmax),
        ]

        for (nf1, nf2, T1, T2) in tpars:

            rn = gp_signals.FourierBasisGP(spectrum=pl, components=nf1, Tspan=T1)
            crn = gp_signals.FourierBasisGP(spectrum=cpl, components=nf2, Tspan=T2)
            s = rn + crn
            rnm = s(self.psr)

            # set up frequencies
            F1, f1 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=nf1, Tspan=T1)
            F2, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=nf2, Tspan=T2)

            # test power spectrum
            p1 = utils.powerlaw(f1, log10_A, gamma)
            p2 = utils.powerlaw(f2, log10_Ac, gammac)
            if T1 == T2:
                nf = max(2 * nf1, 2 * nf2)
                phi = np.zeros(nf)
                F = F1 if nf1 > nf2 else F2
                phi[: 2 * nf1] = p1
                phi[: 2 * nf2] += p2
                F[
                    :,
                ]  # noqa: E231
            else:
                phi = np.concatenate((p1, p2))
                F = np.hstack((F1, F2))

            msg = "Combined red noise PSD incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.all(rnm.get_phi(params) == phi), msg

            msg = "Combined red noise PSD inverse incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

            msg = "Combined red noise Fmat incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.allclose(F, rnm.get_basis(params)), msg

    def test_red_noise_add_backend(self):
        """Test that red noise with backend addition only returns
        independent columns."""
        # set up signals
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        selection = Selection(selections.by_backend)
        cpl = utils.powerlaw(
            log10_A=parameter.Uniform(-18, -12)("log10_Agw"), gamma=parameter.Uniform(1, 7)("gamma_gw")
        )

        # parameters
        log10_As = [-14, -14.4, -15, -14.8]
        gammas = [2.3, 4.4, 1.8, 5.6]
        log10_Ac, gammac = -15.5, 1.33
        params = {
            "B1855+09_red_noise_430_ASP_gamma": gammas[0],
            "B1855+09_red_noise_430_PUPPI_gamma": gammas[1],
            "B1855+09_red_noise_L-wide_ASP_gamma": gammas[2],
            "B1855+09_red_noise_L-wide_PUPPI_gamma": gammas[3],
            "B1855+09_red_noise_430_ASP_log10_A": log10_As[0],
            "B1855+09_red_noise_430_PUPPI_log10_A": log10_As[1],
            "B1855+09_red_noise_L-wide_ASP_log10_A": log10_As[2],
            "B1855+09_red_noise_L-wide_PUPPI_log10_A": log10_As[3],
            "log10_Agw": log10_Ac,
            "gamma_gw": gammac,
        }

        Tmax = self.psr.toas.max() - self.psr.toas.min()
        tpars = [
            (30, 20, Tmax, Tmax),
            (20, 30, Tmax, Tmax),
            (30, 30, Tmax, Tmax),
            (30, 20, Tmax, 1.123 * Tmax),
            (20, 30, Tmax, 1.123 * Tmax),
            (30, 30, 1.123 * Tmax, Tmax),
            (30, 20, None, Tmax),
        ]

        for (nf1, nf2, T1, T2) in tpars:

            rn = gp_signals.FourierBasisGP(spectrum=pl, components=nf1, Tspan=T1, selection=selection)
            crn = gp_signals.FourierBasisGP(spectrum=cpl, components=nf2, Tspan=T2)
            s = rn + crn
            rnm = s(self.psr)

            # get the basis
            bflags = self.psr.backend_flags
            Fmats, fs, phis = [], [], []
            F2, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nf2, Tspan=T2)
            p2 = utils.powerlaw(f2, log10_Ac, gammac)
            for ct, flag in enumerate(np.unique(bflags)):
                mask = bflags == flag
                F1, f1 = utils.createfourierdesignmatrix_red(self.psr.toas[mask], nf1, Tspan=T1)
                Fmats.append(F1)
                fs.append(f1)
                phis.append(utils.powerlaw(f1, log10_As[ct], gammas[ct]))

            Fmats.append(F2)
            phis.append(p2)
            nf = sum(F.shape[1] for F in Fmats)
            F = np.zeros((len(self.psr.toas), nf))
            phi = np.hstack([p for p in phis])
            nftot = 0
            for ct, flag in enumerate(np.unique(bflags)):
                mask = bflags == flag
                nn = Fmats[ct].shape[1]
                F[mask, nftot : nn + nftot] = Fmats[ct]
                nftot += nn
            F[:, -2 * nf2 :] = F2

            msg = "Combined red noise PSD incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.all(rnm.get_phi(params) == phi), msg

            msg = "Combined red noise PSD inverse incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

            msg = "Combined red noise Fmat incorrect "
            msg += "for {} {} {} {}".format(nf1, nf2, T1, T2)
            assert np.allclose(F, rnm.get_basis(params)), msg

    def test_gp_timing_model(self):
        """Test that the timing model signal returns correct values."""
        # set up signal parameter
        ts = gp_signals.TimingModel()
        tm = ts(self.psr)

        # basis matrix test
        M = self.psr.Mmat.copy()
        norm = np.sqrt(np.sum(M ** 2, axis=0))
        M /= norm
        params = {}
        msg = "M matrix incorrect for Timing Model signal."
        assert np.allclose(M, tm.get_basis(params)), msg

        # Jvec test
        phi = np.ones(self.psr.Mmat.shape[1]) * 1e40
        msg = "Prior vector incorrect for Timing Model signal."
        assert np.all(tm.get_phi(params) == phi), msg

        # inverse Jvec test
        msg = "Prior vector inverse incorrect for Timing Model signal."
        assert np.all(tm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "M matrix shape incorrect"
        assert tm.get_basis(params).shape == self.psr.Mmat.shape, msg

    def test_pshift_fourier(self):
        """Test Fourier basis with prescribed phase shifts."""

        # build a SignalCollection with timing model and red noise with phase shifts

        Tspan = self.psr.toas.max() - self.psr.toas.min()
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(0, 7))

        ts = gp_signals.TimingModel()
        rn = gp_signals.FourierBasisGP(pl, components=5, Tspan=Tspan, pseed=parameter.Uniform(0, 32768))

        s = ts + rn
        m = s(self.psr)

        b1 = m.signals[1].get_basis()
        b2 = utils.createfourierdesignmatrix_red(nmodes=5, Tspan=Tspan)("")(self.psr.toas)[0]
        msg = "Fourier bases incorrect (no phase shifts)"
        assert np.all(b1 == b2), msg

        b1 = m.signals[1].get_basis()
        b2 = utils.createfourierdesignmatrix_red(nmodes=5, Tspan=Tspan, pseed=5)("")(self.psr.toas)[0]
        msg = "Fourier bases incorrect (no-parameter call vs phase shift 5)"
        assert not np.all(b1 == b2), msg

        b1 = m.signals[1].get_basis(params={self.psr.name + "_red_noise_pseed": 5})
        b2 = utils.createfourierdesignmatrix_red(nmodes=5, Tspan=Tspan, pseed=5)("")(self.psr.toas)[0]
        msg = "Fourier bases incorrect (phase shift 5)"
        assert np.all(b1 == b2), msg

        b1 = m.signals[1].get_basis(params={self.psr.name + "_red_noise_pseed": 5})
        b2 = utils.createfourierdesignmatrix_red(nmodes=5, Tspan=Tspan)("")(self.psr.toas)[0]
        msg = "Fourier bases incorrect (phase-shift-5 call vs no phase shift)"
        assert not np.all(b1 == b2), msg

    def test_gp_parameter(self):
        """Test GP basis model with parameterized basis."""

        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(0, 7))
        basis_env = utils.createfourierdesignmatrix_env(
            log10_Amp=parameter.Uniform(-10, -5), t0=parameter.Uniform(4.3e9, 5e9), log10_Q=parameter.Uniform(0, 4)
        )

        basis_red = utils.createfourierdesignmatrix_red()

        rn_env = gp_signals.BasisGP(pl, basis_env, name="env")
        rn = gp_signals.BasisGP(pl, basis_red)
        s = rn_env + rn
        m = s(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        log10_A_env, gamma_env = -14.0, 2.5
        log10_Amp, log10_Q, t0 = -7.3, np.log10(345), 55000 * 86400
        params = {
            "B1855+09_log10_A": log10_A,
            "B1855+09_gamma": gamma,
            "B1855+09_env_log10_A": log10_A_env,
            "B1855+09_env_gamma": gamma_env,
            "B1855+09_env_log10_Q": log10_Q,
            "B1855+09_env_log10_Amp": log10_Amp,
            "B1855+09_env_t0": t0,
        }

        # get basis
        Fred, f2_red = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        Fenv, f2_env = utils.createfourierdesignmatrix_env(
            self.psr.toas, nmodes=30, log10_Amp=log10_Amp, log10_Q=log10_Q, t0=t0
        )
        F = np.hstack((Fenv, Fred))
        phi_env = utils.powerlaw(f2_env, log10_A=log10_A_env, gamma=gamma_env)
        phi_red = utils.powerlaw(f2_red, log10_A=log10_A, gamma=gamma)
        phi = np.concatenate((phi_env, phi_red))

        # basis matrix test
        msg = "F matrix incorrect for GP Fourier signal."
        assert np.allclose(F, m.get_basis(params)), msg

        # spectrum test
        msg = "Spectrum incorrect for GP Fourier signal."
        assert np.all(m.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier signal."
        assert np.all(m.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert m.get_basis(params).shape == F.shape, msg

    def test_combine_signals(self):
        """Test for combining different signals."""
        # set up signal parameter
        ecorr = parameter.Uniform(-10, -5)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr)

        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

        log10_sigma = parameter.Uniform(-10, -5)
        log10_lam = parameter.Uniform(np.log10(86400), np.log10(1500 * 86400))
        basis = create_quant_matrix(dt=7 * 86400)
        prior = se_kernel(log10_sigma=log10_sigma, log10_lam=log10_lam)
        se = gp_signals.BasisGP(prior, basis, name="se")

        ts = gp_signals.TimingModel()

        s = ec + rn + ts + se
        m = s(self.psr)

        # parameters
        ecorr = -6.4
        log10_A, gamma = -14.5, 4.33
        log10_lam, log10_sigma = 7.4, -6.4
        params = {
            "B1855+09_basis_ecorr_log10_ecorr": ecorr,
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_se_log10_lam": log10_lam,
            "B1855+09_se_log10_sigma": log10_sigma,
        }

        # combined basis matrix
        U = utils.create_quantization_matrix(self.psr.toas)[0]
        M = self.psr.Mmat.copy()
        norm = np.sqrt(np.sum(M ** 2, axis=0))
        M /= norm
        F, f2 = utils.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        U2, avetoas = create_quant_matrix(self.psr.toas, dt=7 * 86400)
        T = np.hstack((U, F, M, U2))

        # combined prior vector
        jvec = 10 ** (2 * ecorr) * np.ones(U.shape[1])
        phim = np.ones(self.psr.Mmat.shape[1]) * 1e40
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma)
        K = se_kernel(avetoas, log10_lam=log10_lam, log10_sigma=log10_sigma)
        phivec = np.concatenate((jvec, phi, phim))
        phi = sl.block_diag(np.diag(phivec), K)
        phiinv = np.linalg.inv(phi)

        # basis matrix test
        msg = "Basis matrix incorrect for combined signal."
        assert np.allclose(T, m.get_basis(params)), msg

        # Kernal test
        msg = "Prior matrix incorrect for combined signal."
        assert np.allclose(m.get_phi(params), phi), msg

        # inverse Kernel test
        msg = "Prior matrix inverse incorrect for combined signal."
        assert np.allclose(m.get_phiinv(params), phiinv), msg

        # test shape
        msg = "Basis matrix shape incorrect size for combined signal."
        assert m.get_basis(params).shape == T.shape, msg


class TestGPSignalsPint(TestGPSignals):
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
