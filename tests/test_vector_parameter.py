#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_vector_parameter
----------------------------------

Tests for vector parameter functionality
"""


import unittest

import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, signal_base, white_signals
from enterprise.signals.parameter import function
from tests.enterprise_test_data import datadir


@function
def free_spectrum(f, log10_rho=None):
    return np.repeat(10 ** log10_rho, 2)


class TestVectorParameter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_phi(self):
        """Test vector parameter on signal level."""
        # red noise
        nf = 3
        spec = free_spectrum(log10_rho=parameter.Uniform(-20, -10, size=nf))
        rn = gp_signals.FourierBasisGP(spec, components=nf)
        rnm = rn(self.psr)

        # test free-spectrum parameters
        lrho = np.array([-15.5, -16, -14.5])
        params = {"B1855+09_red_noise_log10_rho": lrho}

        # test
        msg = "Phi incorrect"
        assert np.all(np.repeat(10 ** lrho, 2) == rnm.get_phi(params)), msg

        # test signal level parameter names
        msg = "Incorrect parameter names"
        pnames = ["B1855+09_red_noise_log10_rho_0", "B1855+09_red_noise_log10_rho_1", "B1855+09_red_noise_log10_rho_2"]
        assert rnm.param_names == pnames

    def test_vector_parameter_like(self):
        """Test vector parameter in a likelihood"""

        # white noise
        efac = parameter.Uniform(0.5, 2)
        ef = white_signals.MeasurementNoise(efac=efac)

        # red noise
        nf = 3
        spec = free_spectrum(log10_rho=parameter.Uniform(-20, -10, size=nf))
        rn = gp_signals.FourierBasisGP(spec, components=nf)

        # timing model
        tm = gp_signals.TimingModel()

        # combined signal
        s = ef + rn + tm

        # PTA
        pta = signal_base.PTA([s(self.psr)])

        # parameters
        xs = np.hstack(p.sample() for p in pta.params)
        params = {"B1855+09_red_noise_log10_rho": xs[1:], "B1855+09_efac": xs[0]}

        # test log likelihood
        msg = "Likelihoods do not match"
        assert pta.get_lnlikelihood(xs) == pta.get_lnlikelihood(params), msg

        # test log prior
        msg = "Priors do not match"
        assert pta.get_lnprior(xs) == pta.get_lnprior(params), msg

        # test prior value
        prior = 1 / (2 - 0.5) * (1 / 10) ** 3
        msg = "Prior value incorrect."
        assert np.allclose(pta.get_lnprior(xs), np.log(prior)), msg

        # test PTA level parameter names
        pnames = [
            "B1855+09_efac",
            "B1855+09_red_noise_log10_rho_0",
            "B1855+09_red_noise_log10_rho_1",
            "B1855+09_red_noise_log10_rho_2",
        ]
        msg = "Incorrect parameter names"
        assert pta.param_names == pnames


class TestVectorParameterPint(TestVectorParameter):
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
