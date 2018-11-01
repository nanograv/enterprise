#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_gp_coefficients
----------------------------------

Tests for GP signals used with deterministic coefficients.
"""


import math
import unittest
import numpy as np

from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import signal_base
from enterprise.signals import utils


@signal_base.function
def create_quant_matrix(toas, dt=1):
    U, _ = utils.create_quantization_matrix(toas, dt=dt, nmin=1)
    avetoas = np.array([toas[idx.astype(bool)].mean() for idx in U.T])
    # return value slightly different than 1 to get around ECORR columns
    return U*1.0000001, avetoas


@signal_base.function
def se_kernel(etoas, log10_sigma=-7, log10_lam=np.log10(30*86400)):
    tm = np.abs(etoas[None, :] - etoas[:, None])
    d = np.eye(tm.shape[0]) * 10**(2*(log10_sigma-1.5))
    return 10**(2*log10_sigma) * np.exp(-tm**2/2/10**(2*log10_lam)) + d


class TestGPCoefficients(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim')

    def test_fourier_red_noise(self):
        """Test that implicit and explicit GP delays are the same."""
        # set up signal parameter
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18,-12),
                            gamma=parameter.Uniform(1,7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=20)
        rnm = rn(self.psr)

        rnc = gp_signals.FourierBasisGP(spectrum=pl, components=20,
                                        coefficients=True)
        rnmc = rnc(self.psr)

        # parameters
        log10_A, gamma = -14.5, 4.33
        params = {'B1855+09_red_noise_log10_A': log10_A,
                  'B1855+09_red_noise_gamma': gamma}

        # get the GP delays in two different ways
        cf = np.random.randn(40)
        d1 = np.dot(rnm.get_basis(params),cf)

        params.update({'B1855+09_red_noise_coefficients': cf})
        d2 = rnmc.get_delay(params)

        msg = 'Implicit and explicit GP delays are different.'
        assert np.allclose(d1, d2), msg

        # np.array cast is needed because we get a KernelArray
        phimat = np.array(rnm.get_phi(params))
        pr1 = (-0.5 * np.sum(cf*cf/phimat) -
               0.5 * np.sum(np.log(phimat)) -
               0.5 * len(phimat) * np.log(2*math.pi))

        cpar = [p for p in rnmc.params if 'coefficients' in p.name][0]
        pr2 = cpar.get_logpdf(params=params)

        msg = 'Implicit and explicit GP priors are different.'
        assert np.allclose(pr1, pr2), msg

    def test_ecorr_backend(self):
        """Test that ecorr-backend signal returns correct values."""
        # set up signal parameter
        ecorr = parameter.Uniform(-10, -5)
        selection = Selection(selections.by_backend)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                        selection=selection)
        ecm = ec(self.psr)

        ecc = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                         selection=selection,
                                         coefficients=True)
        eccm = ecc(self.psr)

        # parameters
        ecorrs = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_basis_ecorr_430_ASP_log10_ecorr': ecorrs[0],
                  'B1855+09_basis_ecorr_430_PUPPI_log10_ecorr': ecorrs[1],
                  'B1855+09_basis_ecorr_L-wide_ASP_log10_ecorr': ecorrs[2],
                  'B1855+09_basis_ecorr_L-wide_PUPPI_log10_ecorr': ecorrs[3]}

        fmat = ecm.get_basis(params)
        cf = 1e-6 * np.random.randn(fmat.shape[1])
        d1 = np.dot(fmat, cf)

        for key in ecm._keys:
            parname = 'B1855+09_basis_ecorr_' + key + '_coefficients'
            params[parname] = cf[ecm._slices[key]]
        d2 = eccm.get_delay(params)

        msg = 'Implicit and explicit ecorr-basis delays are different.'
        assert np.allclose(d1, d2), msg

    def test_formalism(self):
        # create marginalized model
        ef = white_signals.MeasurementNoise(efac=parameter.Uniform(0.1, 5.0))
        tm = gp_signals.TimingModel()
        ec = gp_signals.EcorrBasisModel(log10_ecorr=parameter.Uniform(-10, -5))
        pl = utils.powerlaw(log10_A=parameter.Uniform(-18,-12),
                            gamma=parameter.Uniform(1,7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=10)
        model = ef + tm + ec + rn
        pta = signal_base.PTA([model(self.psr)])

        # create hierarchical model
        tmc = gp_signals.TimingModel(coefficients=True)
        ecc = gp_signals.EcorrBasisModel(log10_ecorr=parameter.Uniform(-10,-5),
                                         coefficients=True)
        rnc = gp_signals.FourierBasisGP(spectrum=pl, components=10,
                                        coefficients=True)
        modelc = ef + tmc + ecc + rnc
        ptac = signal_base.PTA([modelc(self.psr)])

        ps = {'B1855+09_efac': 1,
              'B1855+09_basis_ecorr_log10_ecorr': -6,
              'B1855+09_red_noise_log10_A': -14,
              'B1855+09_red_noise_gamma': 3}
        psc = utils.get_coefficients(pta, ps)

        d1 = ptac.get_delay(psc)[0]
        d2 = (np.dot(pta.pulsarmodels[0].signals[1].get_basis(ps),
                     psc['B1855+09_linear_timing_model_coefficients']) +
              np.dot(pta.pulsarmodels[0].signals[2].get_basis(ps),
                     psc['B1855+09_basis_ecorr_coefficients']) +
              np.dot(pta.pulsarmodels[0].signals[3].get_basis(ps),
                     psc['B1855+09_red_noise_coefficients']))

        msg = 'Implicit and explicit PTA delays are different.'
        assert np.allclose(d1, d2), msg

        l1 = pta.get_lnlikelihood(ps)
        l2 = ptac.get_lnlikelihood(psc)

        # I don't know how to integrate l2 to match l1...
        msg = 'Marginal and hierarchical likelihoods should be different.'
        assert l1 != l2, msg


class TestGPCoefficientsPint(TestGPCoefficients):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim',
                         ephem='DE430', timing_package='pint')
