#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_gp_priors
----------------------------------

Tests for GP priors and bases.
"""


import unittest
import numpy as np

from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import gp_signals
from enterprise.signals import gp_priors
from enterprise.signals import gp_bases
import scipy.stats


class TestGPSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_turnover_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.turnover(
            log10_A=parameter.Uniform(-18, -12),
            gamma=parameter.Uniform(1, 7),
            lf0=parameter.Uniform(-9, -7.5),
            kappa=parameter.Uniform(2.5, 5),
            beta=parameter.Uniform(0.01, 1),
        )
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma, lf0, kappa, beta = -14.5, 4.33, -8.5, 3, 0.5
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_red_noise_lf0": lf0,
            "B1855+09_red_noise_kappa": kappa,
            "B1855+09_red_noise_beta": beta,
        }

        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for turnover."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.turnover(f2, log10_A=log10_A, gamma=gamma, lf0=lf0, kappa=kappa, beta=beta)
        msg = "Spectrum incorrect for turnover."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for turnover."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_free_spec_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.free_spectrum(log10_rho=parameter.Uniform(-10, -4, size=30))
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)
        # parameters
        rhos = np.random.uniform(-10, -4, size=30)
        params = {"B1855+09_red_noise_log10_rho": rhos}
        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for free spectrum."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.free_spectrum(f2, log10_rho=rhos)

        msg = "Spectrum incorrect for free spectrum."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for free spectrum."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_t_process_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.t_process(
            log10_A=parameter.Uniform(-18, -12),
            gamma=parameter.Uniform(1, 7),
            alphas=gp_priors.InvGamma(alpha=1, gamma=1, size=30),
        )
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)
        # parameters
        alphas = scipy.stats.invgamma.rvs(1, scale=1, size=30)
        log10_A, gamma = -15, 4.33
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_red_noise_alphas": alphas,
        }
        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for free spectrum."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.t_process(f2, log10_A=log10_A, gamma=gamma, alphas=alphas)

        msg = "Spectrum incorrect for free spectrum."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for free spectrum."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_adapt_t_process_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.t_process_adapt(
            log10_A=parameter.Uniform(-18, -12),
            gamma=parameter.Uniform(1, 7),
            alphas_adapt=gp_priors.InvGamma(),
            nfreq=parameter.Uniform(5, 25),
        )
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)
        # parameters
        alphas = scipy.stats.invgamma.rvs(1, scale=1, size=1)
        log10_A, gamma, nfreq = -15, 4.33, 12
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_red_noise_alphas_adapt": alphas,
            "B1855+09_red_noise_nfreq": nfreq,
        }
        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for free spectrum."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.t_process_adapt(f2, log10_A=log10_A, gamma=gamma, alphas_adapt=alphas, nfreq=nfreq)

        msg = "Spectrum incorrect for free spectrum."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for free spectrum."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_turnover_knee_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.turnover_knee(
            log10_A=parameter.Uniform(-18, -12),
            gamma=parameter.Uniform(1, 7),
            lfb=parameter.Uniform(-9, -7.5),
            lfk=parameter.Uniform(-9, -7.5),
            kappa=parameter.Uniform(2.5, 5),
            delta=parameter.Uniform(0.01, 1),
        )
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma, lfb = -14.5, 4.33, -8.5
        lfk, kappa, delta = -8.5, 3, 0.5
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_red_noise_lfb": lfb,
            "B1855+09_red_noise_lfk": lfk,
            "B1855+09_red_noise_kappa": kappa,
            "B1855+09_red_noise_delta": delta,
        }

        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for turnover."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.turnover_knee(f2, log10_A=log10_A, gamma=gamma, lfb=lfb, lfk=lfk, kappa=kappa, delta=delta)
        msg = "Spectrum incorrect for turnover."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for turnover."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_broken_powerlaw_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.broken_powerlaw(
            log10_A=parameter.Uniform(-18, -12),
            gamma=parameter.Uniform(1, 7),
            log10_fb=parameter.Uniform(-9, -7.5),
            kappa=parameter.Uniform(0.1, 1.0),
            delta=parameter.Uniform(0.01, 1),
        )
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma, log10_fb, kappa, delta = -14.5, 4.33, -8.5, 1, 0.5
        params = {
            "B1855+09_red_noise_log10_A": log10_A,
            "B1855+09_red_noise_gamma": gamma,
            "B1855+09_red_noise_log10_fb": log10_fb,
            "B1855+09_red_noise_kappa": kappa,
            "B1855+09_red_noise_delta": delta,
        }

        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_red(self.psr.toas, nmodes=30)
        msg = "F matrix incorrect for turnover."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.broken_powerlaw(f2, log10_A=log10_A, gamma=gamma, log10_fb=log10_fb, kappa=kappa, delta=delta)
        msg = "Spectrum incorrect for turnover."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for turnover."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg

    def test_powerlaw_genmodes_prior(self):
        """Test that red noise signal returns correct values."""
        # set up signal parameter
        pr = gp_priors.powerlaw_genmodes(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        rnm = rn(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        params = {"B1855+09_red_noise_log10_A": log10_A, "B1855+09_red_noise_gamma": gamma}

        # basis matrix test
        F, f2 = gp_bases.createfourierdesignmatrix_chromatic(self.psr.toas, self.psr.freqs, nmodes=30)
        msg = "F matrix incorrect for turnover."
        assert np.allclose(F, rnm.get_basis(params)), msg

        # spectrum test
        phi = gp_priors.powerlaw_genmodes(f2, log10_A=log10_A, gamma=gamma)
        msg = "Spectrum incorrect for turnover."
        assert np.all(rnm.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for turnover."
        assert np.all(rnm.get_phiinv(params) == 1 / phi), msg

        # test shape
        msg = "F matrix shape incorrect"
        assert rnm.get_basis(params).shape == F.shape, msg
