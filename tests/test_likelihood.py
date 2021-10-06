#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_likelihood
----------------------------------

Tests of likelihood module
"""


import unittest

import numpy as np
import scipy.linalg as sl

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, signal_base, utils, white_signals
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
            par = "log10_equad"
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


class TestLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psrs = [
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim"),
            Pulsar(datadir + "/J1909-3744_NANOGrav_9yv1.gls.par", datadir + "/J1909-3744_NANOGrav_9yv1.tim"),
        ]

    def compute_like(self, npsrs=1, inc_corr=False, inc_kernel=False, cholesky_sparse=True, marginalizing_tm=False):
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

        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
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
            s = tm + ef + eq + ec + rn + crn
        else:
            s = tm + ef + eq + ec + rn

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
                nvec0[ind] = efacs[ii][ct] ** 2 * psr.toaerrs[ind] ** 2
                nvec0[ind] += 10 ** (2 * equads[ii][ct]) * np.ones(np.sum(ind))

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
            norm = np.sqrt(np.sum(Mmat ** 2, axis=0))
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

    def test_like_nocorr(self):
        """Test likelihood with no spatial correlations."""
        self.compute_like(npsrs=1)
        self.compute_like(npsrs=2)

    def test_like_corr(self):
        """Test likelihood with spatial correlations."""
        for cholesky_sparse in [True, False]:
            for marginalizing_tm in [True, False]:
                self.compute_like(
                    npsrs=2, inc_corr=True, cholesky_sparse=cholesky_sparse, marginalizing_tm=marginalizing_tm
                )

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
