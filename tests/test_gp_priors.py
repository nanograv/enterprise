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
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import selections
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

    def test_chromatic_fourier_basis_varied_idx(self):
        """Test the set up of variable index chromatic bases and make sure that the caching is the same as no caching"""
        idx = parameter.Uniform(2.5, 7)
        uncached_basis = gp_bases.createfourierdesignmatrix_chromatic(
            self.psr.toas, self.psr.freqs, nmodes=100, idx=idx
        )
        fmat_red, Ffreqs, nus = gp_bases.construct_chromatic_cached_parts(self.psr.toas, self.psr.freqs, nmodes=100)
        cached_basis = gp_bases.createfourierdesignmatrix_chromatic_with_additional_caching(
            fmat_red=fmat_red, Ffreqs=Ffreqs, fref_over_radio_freqs=nus, idx=idx
        )
        pr = gp_priors.powerlaw(log10_A=parameter.Uniform(-18, -11), gamma=parameter.Uniform(1, 7))
        uncached = gp_signals.BasisGP(priorFunction=pr, basisFunction=uncached_basis, name="chrom_gp")
        cached = gp_signals.BasisGP(priorFunction=pr, basisFunction=cached_basis, name="chrom_gp")
        pr = gp_priors.powerlaw_genmodes(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        basis = gp_bases.createfourierdesignmatrix_red(nmodes=30)
        rn = gp_signals.BasisGP(priorFunction=pr, basisFunction=basis, name="red_noise")
        efac = parameter.Normal(1.0, 0.1)
        backend = selections.Selection(selections.by_backend)
        equad = parameter.Uniform(-8.5, -5)
        wn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=backend, name=None)
        mod1 = uncached + rn + wn
        mod2 = cached + rn + wn
        uncached_pta = signal_base.PTA([mod1(self.psr)])
        cached_pta = signal_base.PTA([mod2(self.psr)])

        # check that both of the chromatic bases have the chromatic index as a parameter
        msg = "chromatic index missing from pta parameter list"
        assert "B1855+09_chrom_gp_idx" in uncached_pta.param_names, msg
        assert "B1855+09_chrom_gp_idx" in cached_pta.param_names, msg

        # test to make sure the likelihood evaluations agree for 10 calls
        msg = "the likelihood from cached chromatic basis disagrees with the uncached chroamtic basis likelihood"
        x0 = [np.hstack([p.sample() for p in cached_pta.params]) for _ in range(10)]
        no_cache_lnlike = [uncached_pta.get_lnlikelihood(x0[i]) for i in range(10)]
        cache_lnlike = [cached_pta.get_lnlikelihood(x0[i]) for i in range(10)]
        assert np.all(no_cache_lnlike == cache_lnlike), msg

        # check that both the cached and the uncached basis yield the same basis
        msg = "the cached chromatic basis does not match the uncached chromatic basis"
        assert np.all(uncached_pta.get_basis(params={})[0] == cached_pta.get_basis(params={})[0]), msg
