#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pta
----------------------------------

Tests for common signal and PTA class modules.
"""


# import os
# import pickle
import itertools
import unittest

import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

from .enterprise_test_data import datadir

# note function is now defined in enterprise.signals.parameter


@signal_base.function
def hd_orf_generic(pos1, pos2, a=1.5, b=0.25, c=0.25):
    if np.all(pos1 == pos2):
        return 1
    else:
        xi = 1 - np.dot(pos1, pos2)
        omc2 = (1 - np.cos(xi)) / 2
        return a * omc2 * np.log(omc2) - b * omc2 + c


@signal_base.function
def hd_powerlaw(f, pos1, pos2, log10_A=-15, gamma=4.3):
    return utils.powerlaw(f, log10_A, gamma) * utils.hd_orf(pos1, pos2)


class TestPTASignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        cls.psrs = [
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim"),
            Pulsar(datadir + "/J1909-3744_NANOGrav_9yv1.gls.par", datadir + "/J1909-3744_NANOGrav_9yv1.tim"),
        ]

    def test_parameterized_orf(self):
        T1 = 3.16e8
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        orf = hd_orf_generic(a=parameter.Uniform(0, 5), b=parameter.Uniform(0, 5), c=parameter.Uniform(0, 5))
        rn = gp_signals.FourierBasisGP(spectrum=pl, Tspan=T1, components=30)
        crn = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=orf, components=30, name="gw", Tspan=T1)

        model = rn + crn
        pta = model(self.psrs[0]) + model(self.psrs[1])

        lA1, gamma1 = -13, 1e-15
        lA2, gamma2 = -13.3, 1e-15
        lAc, gammac = -13.1, 1e-15
        a, b, c = 1.9, 0.4, 0.23

        params = {
            "gw_log10_A": lAc,
            "gw_gamma": gammac,
            "gw_a": a,
            "gw_b": b,
            "gw_c": c,
            "B1855+09_red_noise_log10_A": lA1,
            "B1855+09_red_noise_gamma": gamma1,
            "J1909-3744_red_noise_log10_A": lA2,
            "J1909-3744_red_noise_gamma": gamma2,
        }

        phi = pta.get_phi(params)
        phiinv = pta.get_phiinv(params)

        F1, f1 = utils.createfourierdesignmatrix_red(self.psrs[0].toas, nmodes=30, Tspan=T1)
        F2, f2 = utils.createfourierdesignmatrix_red(self.psrs[1].toas, nmodes=30, Tspan=T1)

        msg = "F matrix incorrect"
        assert np.allclose(pta.get_basis(params)[0], F1, rtol=1e-10), msg
        assert np.allclose(pta.get_basis(params)[1], F2, rtol=1e-10), msg

        nftot = 120
        phidiag = np.zeros(nftot)
        phit = np.zeros((nftot, nftot))

        phidiag[:60] = utils.powerlaw(f1, lA1, gamma1)
        phidiag[:60] += utils.powerlaw(f1, lAc, gammac)
        phidiag[60:] = utils.powerlaw(f2, lA2, gamma2)
        phidiag[60:] += utils.powerlaw(f2, lAc, gammac)

        phit[np.diag_indices(nftot)] = phidiag
        orf = hd_orf_generic(self.psrs[0].pos, self.psrs[1].pos, a=a, b=b, c=c)
        spec = utils.powerlaw(f1, log10_A=lAc, gamma=gammac)
        phit[:60, 60:] = np.diag(orf * spec)
        phit[60:, :60] = phit[:60, 60:]

        msg = "{} {}".format(np.diag(phi), np.diag(phit))
        assert np.allclose(phi, phit, rtol=1e-15, atol=1e-17), msg
        msg = "PTA Phi inverse is incorrect {}.".format(params)
        assert np.allclose(phiinv, np.linalg.inv(phit), rtol=1e-15, atol=1e-17), msg

    def test_pta_phiinv_methods(self):
        ef = white_signals.MeasurementNoise(efac=parameter.Uniform(0.1, 5))

        span = np.max(self.psrs[0].toas) - np.min(self.psrs[0].toas)

        pl = utils.powerlaw(log10_A=parameter.Uniform(-16, -13), gamma=parameter.Uniform(1, 7))

        orf = utils.hd_orf()
        vrf = utils.dipole_orf()

        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=span)

        hdrn = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=orf, components=20, Tspan=span, name="gw")

        vrn = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=vrf, components=20, Tspan=span, name="vec")

        vrn2 = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=vrf, components=20, Tspan=span * 1.234, name="vec2")

        # two common processes, sharing basis partially

        model = ef + rn + hdrn  # + vrn

        pta = signal_base.PTA([model(psr) for psr in self.psrs])

        ps = parameter.sample(pta.params)

        phi = pta.get_phi(ps)
        ldp = np.linalg.slogdet(phi)[1]

        inv1, ld1 = pta.get_phiinv(ps, method="cliques", logdet=True)
        inv2, ld2 = pta.get_phiinv(ps, method="partition", logdet=True)
        inv3, ld3 = pta.get_phiinv(ps, method="sparse", logdet=True)
        if not isinstance(inv3, np.ndarray):
            inv3 = inv3.toarray()

        for ld in [ld1, ld2, ld3]:
            msg = "Wrong phi log determinant for two common processes"
            assert np.allclose(ldp, ld, rtol=1e-15, atol=1e-6), msg

        for inv in [inv1, inv2, inv3]:
            msg = "Wrong phi inverse for two common processes"
            assert np.allclose(np.dot(phi, inv), np.eye(phi.shape[0]), rtol=1e-15, atol=1e-6), msg

        for inva, invb in itertools.combinations([inv1, inv2, inv3], 2):
            assert np.allclose(inva, invb)

        # two common processes, no sharing basis

        model = ef + rn + vrn2

        pta = signal_base.PTA([model(psr) for psr in self.psrs])

        ps = parameter.sample(pta.params)

        phi = pta.get_phi(ps)
        ldp = np.linalg.slogdet(phi)[1]

        inv1, ld1 = pta.get_phiinv(ps, method="cliques", logdet=True)
        inv2, ld2 = pta.get_phiinv(ps, method="partition", logdet=True)
        inv3, ld3 = pta.get_phiinv(ps, method="sparse", logdet=True)
        if not isinstance(inv3, np.ndarray):
            inv3 = inv3.toarray()

        for ld in [ld1, ld2, ld3]:
            msg = "Wrong phi log determinant for two common processes"
            assert np.allclose(ldp, ld, rtol=1e-15, atol=1e-6), msg

        for inv in [inv1, inv2, inv3]:
            msg = "Wrong phi inverse for two processes"
            assert np.allclose(np.dot(phi, inv), np.eye(phi.shape[0]), rtol=1e-15, atol=1e-6), msg

        for inva, invb in itertools.combinations([inv1, inv2, inv3], 2):
            assert np.allclose(inva, invb)

        # three common processes, sharing basis partially

        model = ef + rn + hdrn + vrn

        pta = signal_base.PTA([model(psr) for psr in self.psrs])

        ps = parameter.sample(pta.params)

        phi = pta.get_phi(ps)
        ldp = np.linalg.slogdet(phi)[1]

        inv1, ld1 = pta.get_phiinv(ps, method="cliques", logdet=True)
        inv2, ld2 = pta.get_phiinv(ps, method="partition", logdet=True)
        inv3, ld3 = pta.get_phiinv(ps, method="sparse", logdet=True)
        if not isinstance(inv3, np.ndarray):
            inv3 = inv3.toarray()

        for ld in [ld1, ld3]:
            msg = "Wrong phi log determinant for two common processes"
            assert np.allclose(ldp, ld, rtol=1e-15, atol=1e-6), msg

        for inv in [inv1, inv3]:
            msg = "Wrong phi inverse for three common processes"
            assert np.allclose(np.dot(phi, inv), np.eye(phi.shape[0]), rtol=1e-15, atol=1e-6), msg

        for inva, invb in itertools.combinations([inv1, inv3], 2):
            assert np.allclose(inva, invb)

        # four common processes, three sharing basis partially

        model = ef + rn + hdrn + vrn + vrn2

        pta = signal_base.PTA([model(psr) for psr in self.psrs])

        ps = parameter.sample(pta.params)

        phi = pta.get_phi(ps)
        ldp = np.linalg.slogdet(phi)[1]

        inv1, ld1 = pta.get_phiinv(ps, method="cliques", logdet=True)
        inv2, ld2 = pta.get_phiinv(ps, method="partition", logdet=True)
        inv3, ld3 = pta.get_phiinv(ps, method="sparse", logdet=True)
        if not isinstance(inv3, np.ndarray):
            inv3 = inv3.toarray()

        for ld in [ld1, ld3]:
            msg = "Wrong phi log determinant for two common processes"
            assert np.allclose(ldp, ld, rtol=1e-15, atol=1e-6), msg

        for inv in [inv1, inv3]:
            msg = "Wrong phi inverse for four processes"
            assert np.allclose(np.dot(phi, inv), np.eye(phi.shape[0]), rtol=1e-15, atol=1e-6), msg

        for inva, invb in itertools.combinations([inv1, inv3], 2):
            assert np.allclose(inva, invb)

    def test_pta_phi(self):
        T1, T2, T3 = 3.16e8, 3.16e8, 3.16e8
        nf1, nf2, nf3 = 2, 2, 1

        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        orf = utils.hd_orf()
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nf1, Tspan=T1)
        crn = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=orf, components=1, name="gw", Tspan=T3)

        model = rn + crn
        pta = model(self.psrs[0]) + model(self.psrs[1])

        lA1, gamma1 = -13, 1e-15
        lA2, gamma2 = -13.3, 1e-15
        lAc, gammac = -13.1, 1e-15

        params = {
            "gw_log10_A": lAc,
            "gw_gamma": gammac,
            "B1855+09_red_noise_log10_A": lA1,
            "B1855+09_red_noise_gamma": gamma1,
            "J1909-3744_red_noise_log10_A": lA2,
            "J1909-3744_red_noise_gamma": gamma2,
        }

        phi = pta.get_phi(params)
        phiinv = pta.get_phiinv(params)

        T1, T2, T3 = 3.16e8, 3.16e8, 3.16e8
        nf1, nf2, nf3 = 2, 2, 1

        F1, f1 = utils.createfourierdesignmatrix_red(self.psrs[0].toas, nf1, Tspan=T1)
        F2, f2 = utils.createfourierdesignmatrix_red(self.psrs[1].toas, nf2, Tspan=T2)
        F1c, fc = utils.createfourierdesignmatrix_red(self.psrs[0].toas, nf3, Tspan=T3)
        F2c, fc = utils.createfourierdesignmatrix_red(self.psrs[1].toas, nf3, Tspan=T3)

        nftot = 2 * 2 * nf1
        phidiag = np.zeros(nftot)
        phit = np.zeros((nftot, nftot))

        phidiag[:4] = utils.powerlaw(f1, lA1, gamma1)
        phidiag[:2] += utils.powerlaw(fc, lAc, gammac)
        phidiag[4:] = utils.powerlaw(f2, lA2, gamma2)
        phidiag[4:6] += utils.powerlaw(fc, lAc, gammac)

        phit[np.diag_indices(nftot)] = phidiag

        phit[:2, 4:6] = np.diag(hd_powerlaw(fc, self.psrs[0].pos, self.psrs[1].pos, lAc, gammac))
        phit[4:6, :2] = np.diag(hd_powerlaw(fc, self.psrs[0].pos, self.psrs[1].pos, lAc, gammac))

        msg = "{} {}".format(np.diag(phi), np.diag(phit))
        assert np.allclose(phi, phit, rtol=1e-15, atol=1e-17), msg

        msg = "PTA Phi inverse is incorrect {}.".format(params)
        assert np.allclose(phiinv, np.linalg.inv(phit), rtol=1e-15, atol=1e-17), msg

    def test_summary(self):
        """ Test summary table."""
        T1, T3 = 3.16e8, 3.16e8
        nf1 = 30

        pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -12), gamma=parameter.Uniform(1, 7))
        orf = utils.hd_orf()
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nf1, Tspan=T1)
        crn = gp_signals.FourierBasisCommonGP(spectrum=pl, orf=orf, components=1, name="gw", Tspan=T3)

        model = rn + crn
        pta = model(self.psrs[0]) + model(self.psrs[1])
        pta.summary(to_stdout=True)


class TestPTASignalsPint(TestPTASignals):
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
