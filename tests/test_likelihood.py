#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_likelihood
----------------------------------

Tests of likelihood module
"""

import unittest
from types import SimpleNamespace
from unittest import mock
import pytest

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, signal_base, utils, white_signals
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir
from tests.enterprise_test_data import LIBSTEMPO_INSTALLED, PINT_INSTALLED


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
    return 10 ** (2 * log10_sigma) * np.exp(-(tm**2) / 2 / 10 ** (2 * log10_lam)) + d


def get_noise_from_pal2(noisefile):
    psrname = noisefile.split("/")[-1].split("_noise.txt")[0]
    fin = open(noisefile, "r")
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if "efac" in line:
            par = "efac"
            flag = ln[0].split("efac-")[-1]
        elif "equad" in line:
            par = "log10_t2equad"
            flag = ln[0].split("equad-")[-1]
        elif "jitter_q" in line:
            par = "log10_ecorr"
            flag = ln[0].split("jitter_q-")[-1]
        elif "RN-Amplitude" in line:
            par = "red_noise_log10_A"
            flag = ""
        elif "RN-spectral-index" in line:
            par = "red_noise_gamma"
            flag = ""
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = "_".join(name)
        params.update({pname: float(ln[1])})
    return params


class TestSchurComplement(unittest.TestCase):
    def test_scaled_solve(self):
        """Keep the solve and determinant invariant under coefficient rescaling."""

        rng = np.random.default_rng(457)
        basis = rng.normal(size=(12, 4))
        matrix = np.dot(basis.T, basis) + np.eye(4)
        rhs = rng.normal(size=(4, 3))
        expected = sl.solve(matrix, rhs, assume_a="pos")
        expected_logdet = np.linalg.slogdet(matrix)[1]
        for power in [-120, -14, 0, 14, 120]:
            scale = np.array([10.0**power, 1.0, 2.0, 0.3])
            solution, logdet = signal_base.SchurLogLikelihood._solve(
                matrix * scale[:, None] * scale[None, :], rhs * scale[:, None]
            )
            np.testing.assert_allclose(solution * scale[:, None], expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(logdet - 2 * np.log(scale).sum(), expected_logdet, rtol=0, atol=1e-12)

    def test_block_elimination(self):
        """Compare the reduced system with a complete dense factorization."""

        rng = np.random.default_rng(458)
        sizes = [4, 3, 0, 2, 2]
        splits = [
            (np.array([3, 0]), np.array([1, 2])),
            (np.array([1]), np.array([0, 2])),
            (np.array([], dtype=int), np.array([], dtype=int)),
            (np.array([], dtype=int), np.array([0, 1])),
            (np.array([1, 0]), np.array([], dtype=int)),
        ]
        TNrs, TNTs, local_priors, global_common = [], [], [], []
        offset = 0
        for size, (common, local) in zip(sizes, splits):
            if not size:
                TNrs.append(None)
                TNTs.append(None)
                continue
            basis = rng.normal(size=(20, size))
            TNTs.append(np.dot(basis.T, basis))
            TNrs.append(rng.normal(size=size))
            local_priors.append(np.eye(size))
            global_common.extend(offset + common)
            offset += size
        phiinv = sl.block_diag(*local_priors)
        common = np.asarray(global_common)
        cross = rng.normal(size=(len(common), len(common)))
        phiinv[np.ix_(common, common)] += np.dot(cross.T, cross)
        sigma = sl.block_diag(*[t for t in TNTs if t is not None]) + phiinv
        rhs = np.concatenate([d for d in TNrs if d is not None])
        expected_q = np.dot(rhs, sl.solve(sigma, rhs, assume_a="pos"))
        expected_logdet = np.linalg.slogdet(sigma)[1]
        saved_TNTs = [None if t is None else t.copy() for t in TNTs]
        saved_phiinv = phiinv.copy()
        likelihood = signal_base.SchurLogLikelihood(None)

        for precision in [phiinv, sps.csc_matrix(phiinv)]:
            with mock.patch.object(likelihood, "_solve", wraps=likelihood._solve) as solve:
                quadratic, logdet = likelihood._schur(TNrs, TNTs, precision, splits)
            assert [call.args[0].shape for call in solve.call_args_list] == [(2, 2), (2, 2), (2, 2), (5, 5)]
            np.testing.assert_allclose(quadratic, expected_q, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(logdet, expected_logdet, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(phiinv, saved_phiinv)
        for original, current in zip(saved_TNTs, TNTs):
            if original is not None:
                np.testing.assert_array_equal(original, current)

    def test_mixed_prior_fallback(self):
        """Use the reference likelihood when a dense prior crosses the split."""

        @signal_base.function
        def basis(toas, common=False):
            columns = np.column_stack([np.sin(toas), np.cos(toas)])
            return (columns[:, :1], np.zeros(1)) if common else (columns, np.arange(2))

        @signal_base.function
        def prior(labels, common=False):
            return np.ones(len(labels)) if common else np.array([[1.0, 0.4], [0.4, 2.0]])

        rng = np.random.default_rng(459)
        psrs = [
            SimpleNamespace(
                name="mixed" + str(i),
                toas=np.arange(12.0),
                toaerrs=np.ones(12),
                residuals=rng.normal(size=12),
                pos=np.eye(3)[i],
            )
            for i in range(2)
        ]
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        local = gp_signals.BasisGP(prior(), basis(), name="local")
        common = gp_signals.BasisCommonGP(prior(common=True), basis(common=True), utils.hd_orf(), name="gw")
        pta = signal_base.PTA([(white + local + common)(psr) for psr in psrs])
        expected = pta.get_lnlikelihood({})
        with mock.patch.object(signal_base.LogLikelihood, "__call__", return_value=expected) as fallback:
            actual = signal_base.SchurLogLikelihood(pta)({})
        fallback.assert_called_once_with({}, phiinv_method="cliques")
        assert actual == expected

    def test_common_basis_wider_than_toas(self):
        """A finite common prior supports more Fourier columns than TOAs."""

        rng = np.random.default_rng(460)
        toas = np.sort(rng.uniform(0, 10 * 365.25 * 86400, 40))
        psrs = [
            SimpleNamespace(
                name="wide" + str(i),
                toas=toas,
                toaerrs=np.full(40, 1e-6),
                residuals=rng.normal(size=40) * 1e-6,
                pos=np.eye(3)[i],
            )
            for i in range(2)
        ]
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        spectrum = utils.powerlaw(log10_A=parameter.Constant(-15.0), gamma=parameter.Constant(4.33))
        common = gp_signals.FourierBasisCommonGP(spectrum, utils.hd_orf(), components=25, Tspan=np.ptp(toas))
        pta = signal_base.PTA([(white + common)(psr) for psr in psrs])
        expected = pta.get_lnlikelihood({})
        with mock.patch.object(signal_base.LogLikelihood, "__call__", side_effect=AssertionError("fallback")):
            actual = signal_base.SchurLogLikelihood(pta)({})
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_equivalent_basis_priors(self):
        """Equivalent basis/prior scalings must give the same Gaussian likelihood."""

        @signal_base.function
        def local_basis(toas, scale=1.0):
            scales = np.array([1.0, scale])
            return np.eye(len(toas))[:, 1:3] * scales, scales

        @signal_base.function
        def local_prior(scales, dense=False):
            if dense:
                return np.array([[1.0, 0.2], [0.2, 2.0]]) / np.outer(scales, scales)
            return scales**-2

        @signal_base.function
        def common_basis(toas):
            return np.eye(len(toas))[:, :1], np.ones(1)

        @signal_base.function
        def common_prior(labels):
            return np.ones_like(labels)

        psrs = [
            SimpleNamespace(
                name="scaled" + str(i),
                toas=np.arange(6.0),
                toaerrs=np.ones(6),
                residuals=np.arange(1.0, 7.0),
                pos=np.eye(3)[i],
            )
            for i in range(2)
        ]
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        common = gp_signals.BasisCommonGP(common_prior(), common_basis(), utils.hd_orf(), name="gw")
        for dense in [False, True]:
            reference = None
            for scale in [1.0, 1e-10, 1e-12, 1e-14, 1e14]:
                local = gp_signals.BasisGP(local_prior(dense=dense), local_basis(scale=scale), name="local")
                pta = signal_base.PTA([(white + local + common)(psr) for psr in psrs])
                basis = sl.block_diag(*pta.get_basis({}))
                covariance = np.eye(12) + np.dot(np.dot(basis, pta.get_phi({})), basis.T)
                factor = sl.cho_factor(covariance)
                residual = np.concatenate(pta.get_residuals())
                expected = -0.5 * (
                    np.dot(residual, sl.cho_solve(factor, residual))
                    + 2 * np.log(np.diag(factor[0])).sum()
                    + residual.size * np.log(2 * np.pi)
                )
                with mock.patch.object(signal_base.LogLikelihood, "__call__", side_effect=AssertionError("fallback")):
                    actual = signal_base.SchurLogLikelihood(pta)({})
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
                if reference is None:
                    reference = actual
                np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-10)

    def test_weak_red_noise(self):
        """Keep finite-prior directions when local red noise becomes very weak."""

        rng = np.random.default_rng(15457)
        toas = np.sort(rng.uniform(0, 10 * 365.25 * 86400, 400))
        psrs = [
            SimpleNamespace(
                name="weak" + str(i),
                toas=toas,
                toaerrs=np.full(400, 1e-6),
                residuals=rng.normal(size=400) * 1e-6,
                pos=np.eye(3)[i],
                Mmat=np.column_stack([np.ones(400), toas, toas**2]),
            )
            for i in range(2)
        ]
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        timing = gp_signals.TimingModel()
        spectrum = utils.powerlaw(log10_A=parameter.Constant(-15.0), gamma=parameter.Constant(4.33))
        common = gp_signals.FourierBasisCommonGP(spectrum, utils.hd_orf(), components=5, Tspan=np.ptp(toas), name="gw")
        red_spectrum = utils.powerlaw(log10_A=parameter.Uniform(-22, -14), gamma=parameter.Constant(7.0))
        red = gp_signals.FourierBasisGP(red_spectrum, components=100, Tspan=np.ptp(toas), name="rn")
        pta = signal_base.PTA([(timing + white + red + common)(psr) for psr in psrs])
        schur = signal_base.SchurLogLikelihood(pta)
        for amplitude in [-16.0, -18.0, -20.0, -22.0]:
            params = {p.name: amplitude for p in pta.params}
            expected = pta.get_lnlikelihood(params)
            with mock.patch.object(signal_base.LogLikelihood, "__call__", side_effect=AssertionError("fallback")):
                actual = schur(params)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


class TestLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psrs = [
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.t2.feather"),
            Pulsar(datadir + "/J1909-3744_NANOGrav_9yv1.t2.feather"),
        ]

    def compute_like(
        self, npsrs=1, inc_corr=False, inc_kernel=False, cholesky_sparse=True, marginalizing_tm=False, schur=False
    ):
        # get parameters from PAL2 style noise files
        params = get_noise_from_pal2(datadir + "/B1855+09_noise.txt")
        params2 = get_noise_from_pal2(datadir + "/J1909-3744_noise.txt")
        params.update(params2)

        psrs = self.psrs if npsrs == 2 else [self.psrs[0]]

        if inc_corr:
            params.update({"GW_gamma": 4.33, "GW_log10_A": -15.0})

        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in psrs]
        tmax = [p.toas.max() for p in psrs]
        Tspan = np.max(tmax) - np.min(tmin)

        # setup basic model
        efac = parameter.Constant()
        equad = parameter.Constant()
        ecorr = parameter.Constant()
        log10_A = parameter.Constant()
        gamma = parameter.Constant()

        selection = Selection(selections.by_backend)

        ms = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl)

        orf = utils.hd_orf()
        crn = gp_signals.FourierBasisCommonGP(pl, orf, components=20, name="GW", Tspan=Tspan)

        if marginalizing_tm:
            tm = gp_signals.MarginalizingTimingModel()
        else:
            tm = gp_signals.TimingModel()

        log10_sigma = parameter.Uniform(-10, -5)
        log10_lam = parameter.Uniform(np.log10(86400), np.log10(1500 * 86400))
        basis = create_quant_matrix(dt=7 * 86400)
        prior = se_kernel(log10_sigma=log10_sigma, log10_lam=log10_lam)
        se = gp_signals.BasisGP(prior, basis, name="se")

        # set up kernel stuff
        if isinstance(inc_kernel, bool):
            inc_kernel = [inc_kernel] * npsrs

        if inc_corr:
            s = tm + ms + ec + rn + crn
        else:
            s = tm + ms + ec + rn

        models = []
        for ik, psr in zip(inc_kernel, psrs):
            snew = s + se if ik else s
            models.append(snew(psr))

        if cholesky_sparse:
            like = signal_base.LogLikelihood
        else:
            like = signal_base.LogLikelihoodDenseCholesky

        pta = signal_base.PTA(models, lnlikelihood=like)

        # set parameters
        pta.set_default_params(params)

        # SE kernel parameters
        log10_sigmas, log10_lams = [-7.0, -6.5], [7.0, 6.5]
        params.update(
            {
                "B1855+09_se_log10_lam": log10_lams[0],
                "B1855+09_se_log10_sigma": log10_sigmas[0],
                "J1909-3744_se_log10_lam": log10_lams[1],
                "J1909-3744_se_log10_sigma": log10_sigmas[1],
            }
        )

        # get parameters
        efacs, equads, ecorrs, log10_A, gamma = [], [], [], [], []
        lsig, llam = [], []
        for pname in [p.name for p in psrs]:
            efacs.append([params[key] for key in sorted(params.keys()) if "efac" in key and pname in key])
            equads.append([params[key] for key in sorted(params.keys()) if "equad" in key and pname in key])
            ecorrs.append([params[key] for key in sorted(params.keys()) if "ecorr" in key and pname in key])
            log10_A.append(params["{}_red_noise_log10_A".format(pname)])
            gamma.append(params["{}_red_noise_gamma".format(pname)])
            lsig.append(params["{}_se_log10_sigma".format(pname)])
            llam.append(params["{}_se_log10_lam".format(pname)])
        GW_gamma = 4.33
        GW_log10_A = -15.0

        # correct value
        tflags = [sorted(list(np.unique(p.backend_flags))) for p in psrs]
        cfs, logdets, phis, Ts = [], [], [], []
        for ii, (ik, psr, flags) in enumerate(zip(inc_kernel, psrs, tflags)):
            nvec0 = np.zeros_like(psr.toas)
            for ct, flag in enumerate(flags):
                ind = psr.backend_flags == flag
                nvec0[ind] = efacs[ii][ct] ** 2 * (
                    psr.toaerrs[ind] ** 2 + 10 ** (2 * equads[ii][ct]) * np.ones(np.sum(ind))
                )

            # get the basis
            bflags = psr.backend_flags
            Umats = []
            for flag in np.unique(bflags):
                mask = bflags == flag
                Umats.append(utils.create_quantization_matrix(psr.toas[mask])[0])
            nepoch = sum(U.shape[1] for U in Umats)
            U = np.zeros((len(psr.toas), nepoch))
            jvec = np.zeros(nepoch)
            netot = 0
            for ct, flag in enumerate(np.unique(bflags)):
                mask = bflags == flag
                nn = Umats[ct].shape[1]
                U[mask, netot : nn + netot] = Umats[ct]
                jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ii][ct])
                netot += nn

            # get covariance matrix
            cov = np.diag(nvec0) + np.dot(U * jvec[None, :], U.T)
            cf = sl.cho_factor(cov)
            logdet = np.sum(2 * np.log(np.diag(cf[0])))
            cfs.append(cf)
            logdets.append(logdet)

            F, f2 = utils.createfourierdesignmatrix_red(psr.toas, nmodes=20, Tspan=Tspan)
            Mmat = psr.Mmat.copy()
            norm = np.sqrt(np.sum(Mmat**2, axis=0))
            Mmat /= norm
            U2, avetoas = create_quant_matrix(psr.toas, dt=7 * 86400)
            if ik:
                T = np.hstack((F, Mmat, U2))
            else:
                T = np.hstack((F, Mmat))
            Ts.append(T)
            phi = utils.powerlaw(f2, log10_A=log10_A[ii], gamma=gamma[ii])
            if inc_corr:
                phigw = utils.powerlaw(f2, log10_A=GW_log10_A, gamma=GW_gamma)
            else:
                phigw = np.zeros(40)
            K = se_kernel(avetoas, log10_sigma=log10_sigmas[ii], log10_lam=log10_lams[ii])
            k = np.diag(np.concatenate((phi + phigw, np.ones(Mmat.shape[1]) * 1e40)))
            if ik:
                k = sl.block_diag(k, K)
            phis.append(k)

        # manually compute loglike
        loglike = 0
        TNrs, TNTs = [], []
        for ct, psr in enumerate(psrs):
            TNrs.append(np.dot(Ts[ct].T, sl.cho_solve(cfs[ct], psr.residuals)))
            TNTs.append(np.dot(Ts[ct].T, sl.cho_solve(cfs[ct], Ts[ct])))
            loglike += -0.5 * (np.dot(psr.residuals, sl.cho_solve(cfs[ct], psr.residuals)) + logdets[ct])
            loglike += -0.5 * len(psr.residuals) * np.log(2 * np.pi)

        TNr = np.concatenate(TNrs)
        phi = sl.block_diag(*phis)

        if inc_corr:
            hd = utils.hd_orf(psrs[0].pos, psrs[1].pos)
            phi[len(phis[0]) : len(phis[0]) + 40, :40] = np.diag(phigw * hd)
            phi[:40, len(phis[0]) : len(phis[0]) + 40] = np.diag(phigw * hd)

        cf = sl.cho_factor(phi)
        phiinv = sl.cho_solve(cf, np.eye(phi.shape[0]))
        logdetphi = np.sum(2 * np.log(np.diag(cf[0])))
        Sigma = sl.block_diag(*TNTs) + phiinv

        cf = sl.cho_factor(Sigma)
        expval = sl.cho_solve(cf, TNr)
        logdetsigma = np.sum(2 * np.log(np.diag(cf[0])))

        loglike -= 0.5 * (logdetphi + logdetsigma)
        loglike += 0.5 * np.dot(TNr, expval)

        method = ["partition", "sparse", "cliques"]
        for mth in method:
            eloglike = pta.get_lnlikelihood(params, phiinv_method=mth)
            msg = "Incorrect like for npsr={}, phiinv={}, csparse={}, mtm={}".format(
                npsrs, mth, cholesky_sparse, marginalizing_tm
            )
            assert np.allclose(eloglike, loglike), msg

        if schur:
            schur_pta = signal_base.PTA(models, lnlikelihood=signal_base.SchurLogLikelihood)
            methods = ["partition", "sparse", "cliques"]
            for method in methods:
                expected = pta.get_lnlikelihood(params, phiinv_method=method)
                if inc_corr:
                    with mock.patch.object(
                        signal_base.LogLikelihood,
                        "__call__",
                        side_effect=AssertionError("SchurLogLikelihood unexpectedly used LogLikelihood."),
                    ):
                        schur_loglike = schur_pta.get_lnlikelihood(params, phiinv_method=method)
                else:
                    schur_loglike = schur_pta.get_lnlikelihood(params, phiinv_method=method)
                if inc_corr:
                    assert np.isscalar(schur_loglike)
                np.testing.assert_allclose(schur_loglike, expected, rtol=0, atol=1e-4)
            xs = (
                np.hstack([np.atleast_1d(params[p.name]) for p in schur_pta.params])
                if schur_pta.params
                else np.empty(0)
            )
            np.testing.assert_allclose(schur_pta.get_lnlikelihood(xs), expected, rtol=0, atol=1e-4)

    def test_like_nocorr(self):
        """Test likelihood with no spatial correlations."""
        self.compute_like(npsrs=1)
        self.compute_like(npsrs=2)

    def test_like_marginalizing_tm_idx_exclude(self):
        """Test for marginalizing timing model with excluded columns."""
        psr = self.psrs[0]

        tm = gp_signals.MarginalizingTimingModel(idx_exclude=[0])
        mn = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        pta = signal_base.PTA([(tm + mn)(psr)])

        ll = pta.get_lnlikelihood({})
        assert np.isfinite(ll), "Likelihood should be finite with idx_exclude timing model"

    def test_like_corr(self):
        """Test likelihood with spatial correlations."""
        for cholesky_sparse in [True, False]:
            for marginalizing_tm in [True, False]:
                self.compute_like(
                    npsrs=2, inc_corr=True, cholesky_sparse=cholesky_sparse, marginalizing_tm=marginalizing_tm
                )

    def test_like_corr_schur(self):
        """Test the Schur likelihood with spatial correlations."""

        self.compute_like(npsrs=2, inc_corr=True, inc_kernel=True, schur=True)
        self.compute_like(npsrs=2, inc_corr=True, inc_kernel=True, marginalizing_tm=True, schur=True)
        self.compute_like(npsrs=2, marginalizing_tm=True, schur=True)

    def test_like_corr_schur_with_white_noise_only_pulsar(self):
        """Test a correlated model with one pulsar that has no basis."""

        psr3 = Pulsar(datadir + "/B1937+21_NANOGrav_9yv1.t2.feather")
        Tspan = max(p.toas.max() for p in self.psrs) - min(p.toas.min() for p in self.psrs)
        mn = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        eq = white_signals.TNEquadNoise(log10_tnequad=parameter.Constant(-20.0))
        pl = utils.powerlaw(log10_A=parameter.Constant(-15.0), gamma=parameter.Constant(4.33))
        crn = gp_signals.FourierBasisCommonGP(pl, utils.hd_orf(), components=2, name="GW", Tspan=Tspan)

        correlated = [(mn + eq + crn)(psr) for psr in self.psrs]
        white_only = (mn + eq)(psr3)
        corr_pta = signal_base.PTA(correlated, lnlikelihood=signal_base.SchurLogLikelihood)
        white_pta = signal_base.PTA([white_only])
        full_pta = signal_base.PTA(correlated + [white_only], lnlikelihood=signal_base.SchurLogLikelihood)

        expected = corr_pta.get_lnlikelihood({}) + white_pta.get_lnlikelihood({})
        np.testing.assert_allclose(full_pta.get_lnlikelihood({}), expected, rtol=1e-12, atol=1e-7)

    def test_like_schur_multiple_common_signals(self):
        """Test the Schur likelihood with multiple common signals."""

        Tspan = max(p.toas.max() for p in self.psrs) - min(p.toas.min() for p in self.psrs)
        timing = gp_signals.TimingModel()
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        hd_spectrum = utils.powerlaw(log10_A=parameter.Constant(-15.0), gamma=parameter.Constant(4.33))
        monopole_spectrum = utils.powerlaw(log10_A=parameter.Constant(-16.0), gamma=parameter.Constant(3.0))
        hd = gp_signals.FourierBasisCommonGP(hd_spectrum, utils.hd_orf(), components=2, Tspan=Tspan, name="gw")
        monopole = gp_signals.FourierBasisCommonGP(
            monopole_spectrum,
            utils.monopole_orf(),
            components=3,
            Tspan=Tspan,
            name="monopole",
        )
        models = [(timing + white + hd + monopole)(psr) for psr in self.psrs]

        hd_only_model = (timing + white + hd)(self.psrs[1])
        first_pta = signal_base.PTA(
            [models[0], hd_only_model],
            lnlikelihood=signal_base.SchurLogLikelihood,
        )
        first_pta.get_lnlikelihood({})

        standard = signal_base.PTA(models)
        schur = signal_base.PTA(models, lnlikelihood=signal_base.SchurLogLikelihood)
        assert len(schur._commonsignals) == 2
        expected = standard.get_lnlikelihood({})
        with mock.patch.object(
            signal_base.LogLikelihood,
            "__call__",
            side_effect=AssertionError("SchurLogLikelihood unexpectedly used LogLikelihood."),
        ):
            actual = schur.get_lnlikelihood({})
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_like_schur_ecorr(self):
        """Test Schur with all ECORR representations and timing-model paths."""

        Tspan = max(p.toas.max() for p in self.psrs) - min(p.toas.min() for p in self.psrs)
        white = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
        spectrum = utils.powerlaw(log10_A=parameter.Constant(-15.0), gamma=parameter.Constant(4.33))
        common = gp_signals.FourierBasisCommonGP(spectrum, utils.hd_orf(), components=2, Tspan=Tspan, name="gw")
        for timing in [gp_signals.TimingModel(use_svd=True), gp_signals.MarginalizingTimingModel(use_svd=True)]:
            for method in ["sparse", "block", "sherman-morrison", "fast-sherman-morrison"]:
                ecorr = white_signals.EcorrKernelNoise(log10_ecorr=parameter.Constant(-8.0), method=method)
                models = [(timing + white + ecorr + common)(psr) for psr in self.psrs]
                standard = signal_base.PTA(models)
                schur = signal_base.PTA(models, lnlikelihood=signal_base.SchurLogLikelihood)
                # BlockMatrix can return a 1-by-1 quadratic form instead of a scalar.
                white_terms = [tuple(np.asarray(v).item() for v in terms) for terms in standard.get_rNr_logdet({})]
                with mock.patch.object(standard, "get_rNr_logdet", return_value=white_terms):
                    expected = standard.get_lnlikelihood({})
                with mock.patch.object(signal_base.LogLikelihood, "__call__", side_effect=AssertionError("fallback")):
                    actual = schur.get_lnlikelihood({})
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, err_msg=method)

    def test_like_nocorr_kernel(self):
        """Test likelihood with no spatial correlations and kernel."""
        self.compute_like(npsrs=1, inc_kernel=True)
        self.compute_like(npsrs=2, inc_kernel=True)

    def test_like_corr_kernel(self):
        """Test likelihood with spatial correlations and kernel."""
        self.compute_like(npsrs=2, inc_corr=True, inc_kernel=True, cholesky_sparse=True)
        self.compute_like(npsrs=2, inc_corr=True, inc_kernel=True, cholesky_sparse=False)

    def test_like_nocorr_one_kernel(self):
        """Test likelihood with no spatial correlations and one kernel."""
        self.compute_like(npsrs=2, inc_kernel=[True, False])

    def test_like_corr_one_kernel(self):
        """Test likelihood with spatial correlations and one kernel."""
        self.compute_like(npsrs=2, inc_corr=True, inc_kernel=[True, False])

    def test_compare_ecorr_likelihood(self):
        """Compare basis and kernel ecorr methods."""

        selection = Selection(selections.nanograv_backends)
        ef = white_signals.MeasurementNoise()
        ec = white_signals.EcorrKernelNoise(selection=selection)
        ec2 = gp_signals.EcorrBasisModel(selection=selection)
        tm = gp_signals.TimingModel()
        m = ef + ec + tm
        m2 = ef + ec2 + tm

        pta1 = signal_base.PTA([m(p) for p in self.psrs])
        pta2 = signal_base.PTA([m2(p) for p in self.psrs])

        params = parameter.sample(pta1.params)
        l1 = pta1.get_lnlikelihood(params)

        # need to translate some names for EcorrBasis
        basis_params = {}
        for parname, parval in params.items():
            if "log10_ecorr" in parname:
                toks = parname.split("_")
                basisname = toks[0] + "_basis_ecorr_" + "_".join(toks[1:])
                basis_params[basisname] = parval
        params.update(basis_params)
        l2 = pta2.get_lnlikelihood(params)

        msg = "Likelihood mismatch between ECORR methods"
        assert np.allclose(l1, l2), msg

    def test_like_sparse_cache(self):
        """Test likelihood with sparse Cholesky caching"""

        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in self.psrs]
        tmax = [p.toas.max() for p in self.psrs]
        Tspan = np.max(tmax) - np.min(tmin)

        # setup basic model
        efac = parameter.Constant(1.0)
        log10_A = parameter.Constant(-15.0)
        gamma = parameter.Constant(4.33)

        ef = white_signals.MeasurementNoise(efac)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)

        orf = utils.hd_orf()
        crn = gp_signals.FourierBasisCommonGP(pl, orf, components=20, name="GW", Tspan=Tspan)

        tm = gp_signals.TimingModel()
        m = tm + ef + crn

        # Two identical arrays that we'll compare with two sets of parameters
        pta1 = signal_base.PTA([m(p) for p in self.psrs])
        pta2 = signal_base.PTA([m(p) for p in self.psrs])

        params_init = parameter.sample(pta1.params)
        params_check = parameter.sample(pta1.params)

        # First call for pta1 only initializes the sparse decomposition. Second one uses it
        _ = pta1.get_lnlikelihood(params_init)
        l1 = pta1.get_lnlikelihood(params_check)
        l2 = pta2.get_lnlikelihood(params_check)

        msg = "Likelihood mismatch between sparse Cholesky full & inplace"
        assert np.allclose(l1, l2), msg


@pytest.mark.skipif(not PINT_INSTALLED, reason="Skipping tests that require PINT because it isn't installed")
class TestLikelihoodPint(TestLikelihood):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psrs = [
            Pulsar(
                datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
                datadir + "/B1855+09_NANOGrav_9yv1.tim",
                ephem="DE430",
                timing_package="pint",
            ),
            Pulsar(
                datadir + "/J1909-3744_NANOGrav_9yv1.gls.par",
                datadir + "/J1909-3744_NANOGrav_9yv1.tim",
                ephem="DE430",
                timing_package="pint",
            ),
        ]


@pytest.mark.skipif(not LIBSTEMPO_INSTALLED, reason="Skipping tests that require libstempo because it isn't installed")
class TestLikelihoodTempo2(TestLikelihood):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psrs = [
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim"),
            Pulsar(datadir + "/J1909-3744_NANOGrav_9yv1.gls.par", datadir + "/J1909-3744_NANOGrav_9yv1.tim"),
        ]
