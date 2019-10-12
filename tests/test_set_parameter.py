#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_set_parameter
----------------------------------

Tests of setting constant parameters
"""


import unittest

import numpy as np
import scipy.linalg as sl

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, utils, white_signals
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


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


class TestSetParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psrs = [
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim"),
            Pulsar(datadir + "/J1909-3744_NANOGrav_9yv1.gls.par", datadir + "/J1909-3744_NANOGrav_9yv1.tim"),
        ]

    def test_single_pulsar(self):

        # get parameters from PAL2 style noise files
        params = get_noise_from_pal2(datadir + "/B1855+09_noise.txt")

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

        s = ef + eq + ec + rn
        m = s(self.psrs[0])

        # set parameters
        m.set_default_params(params)

        # get parameters
        efacs = [params[key] for key in sorted(params.keys()) if "efac" in key]
        equads = [params[key] for key in sorted(params.keys()) if "equad" in key]
        ecorrs = [params[key] for key in sorted(params.keys()) if "ecorr" in key]
        log10_A = params["B1855+09_red_noise_log10_A"]
        gamma = params["B1855+09_red_noise_gamma"]

        # correct value
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        nvec0 = np.zeros_like(self.psrs[0].toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psrs[0].backend_flags
            nvec0[ind] = efacs[ct] ** 2 * self.psrs[0].toaerrs[ind] ** 2
            nvec0[ind] += 10 ** (2 * equads[ct]) * np.ones(np.sum(ind))

        # get the basis
        bflags = self.psrs[0].backend_flags
        Umats = []
        for flag in np.unique(bflags):
            mask = bflags == flag
            Umats.append(utils.create_quantization_matrix(self.psrs[0].toas[mask])[0])
        nepoch = sum(U.shape[1] for U in Umats)
        U = np.zeros((len(self.psrs[0].toas), nepoch))
        jvec = np.zeros(nepoch)
        netot = 0
        for ct, flag in enumerate(np.unique(bflags)):
            mask = bflags == flag
            nn = Umats[ct].shape[1]
            U[mask, netot : nn + netot] = Umats[ct]
            jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ct])
            netot += nn

        # get covariance matrix
        cov = np.diag(nvec0) + np.dot(U * jvec[None, :], U.T)
        cf = sl.cho_factor(cov)
        logdet = np.sum(2 * np.log(np.diag(cf[0])))

        # test
        msg = "EFAC/ECORR logdet incorrect."
        N = m.get_ndiag(params)
        assert np.allclose(N.solve(self.psrs[0].residuals, logdet=True)[1], logdet, rtol=1e-10), msg

        msg = "EFAC/ECORR D1 solve incorrect."
        assert np.allclose(N.solve(self.psrs[0].residuals), sl.cho_solve(cf, self.psrs[0].residuals), rtol=1e-10), msg

        msg = "EFAC/ECORR 1D1 solve incorrect."
        assert np.allclose(
            N.solve(self.psrs[0].residuals, left_array=self.psrs[0].residuals),
            np.dot(self.psrs[0].residuals, sl.cho_solve(cf, self.psrs[0].residuals)),
            rtol=1e-10,
        ), msg

        msg = "EFAC/ECORR 2D1 solve incorrect."
        T = m.get_basis(params)
        assert np.allclose(
            N.solve(self.psrs[0].residuals, left_array=T),
            np.dot(T.T, sl.cho_solve(cf, self.psrs[0].residuals)),
            rtol=1e-10,
        ), msg

        msg = "EFAC/ECORR 2D2 solve incorrect."
        assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, sl.cho_solve(cf, T)), rtol=1e-10), msg

        F, f2 = utils.createfourierdesignmatrix_red(self.psrs[0].toas, nmodes=20)

        # spectrum test
        phi = utils.powerlaw(f2, log10_A=log10_A, gamma=gamma)
        msg = "Spectrum incorrect for GP Fourier signal."
        assert np.all(m.get_phi(params) == phi), msg

        # inverse spectrum test
        msg = "Spectrum inverse incorrect for GP Fourier signal."
        assert np.all(m.get_phiinv(params) == 1 / phi), msg

    def test_pta(self):

        # get parameters from PAL2 style noise files
        params = get_noise_from_pal2(datadir + "/B1855+09_noise.txt")
        params2 = get_noise_from_pal2(datadir + "/J1909-3744_noise.txt")
        params.update(params2)

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

        s = ef + eq + ec + rn
        pta = s(self.psrs[0]) + s(self.psrs[1])

        # set parameters
        pta.set_default_params(params)

        # get parameters
        efacs, equads, ecorrs, log10_A, gamma = [], [], [], [], []
        for pname in [p.name for p in self.psrs]:
            efacs.append([params[key] for key in sorted(params.keys()) if "efac" in key and pname in key])
            equads.append([params[key] for key in sorted(params.keys()) if "equad" in key and pname in key])
            ecorrs.append([params[key] for key in sorted(params.keys()) if "ecorr" in key and pname in key])
            log10_A.append(params["{}_red_noise_log10_A".format(pname)])
            gamma.append(params["{}_red_noise_gamma".format(pname)])

        # correct value
        tflags = [sorted(list(np.unique(p.backend_flags))) for p in self.psrs]
        cfs, logdets, phis = [], [], []
        for ii, (psr, flags) in enumerate(zip(self.psrs, tflags)):
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

            F, f2 = utils.createfourierdesignmatrix_red(psr.toas, nmodes=20)
            phi = utils.powerlaw(f2, log10_A=log10_A[ii], gamma=gamma[ii])
            phis.append(phi)

        # tests
        Ns = pta.get_ndiag(params)
        pphis = pta.get_phi(params)
        pphiinvs = pta.get_phiinv(params)
        Ts = pta.get_basis(params)
        zipped = zip(logdets, cfs, phis, self.psrs, Ns, pphis, pphiinvs, Ts)
        for logdet, cf, phi, psr, N, pphi, pphiinv, T in zipped:
            msg = "EFAC/ECORR logdet incorrect."
            assert np.allclose(N.solve(psr.residuals, logdet=True)[1], logdet, rtol=1e-10), msg

            msg = "EFAC/ECORR D1 solve incorrect."
            assert np.allclose(N.solve(psr.residuals), sl.cho_solve(cf, psr.residuals), rtol=1e-10), msg

            msg = "EFAC/ECORR 1D1 solve incorrect."
            assert np.allclose(
                N.solve(psr.residuals, left_array=psr.residuals),
                np.dot(psr.residuals, sl.cho_solve(cf, psr.residuals)),
                rtol=1e-10,
            ), msg

            msg = "EFAC/ECORR 2D1 solve incorrect."
            assert np.allclose(
                N.solve(psr.residuals, left_array=T), np.dot(T.T, sl.cho_solve(cf, psr.residuals)), rtol=1e-10
            ), msg

            msg = "EFAC/ECORR 2D2 solve incorrect."
            assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, sl.cho_solve(cf, T)), rtol=1e-10), msg

            # spectrum test
            msg = "Spectrum incorrect for GP Fourier signal."
            assert np.all(pphi == phi), msg

            # inverse spectrum test
            msg = "Spectrum inverse incorrect for GP Fourier signal."
            assert np.all(pphiinv == 1 / phi), msg


class TestSetParametersPint(TestSetParameters):
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
