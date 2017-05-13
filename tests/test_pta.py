#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pta
----------------------------------

Tests for common signal and PTA class modules.
"""


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import unittest
import numpy as np

from enterprise.pulsar import Pulsar

import enterprise.signals.parameter as parameter
import enterprise.signals.signal_base as signal_base
import enterprise.signals.gp_signals as gp_signals
from enterprise.signals import utils

from tests.enterprise_test_data import datadir


def hd_orf(pos1, pos2):
    xi = 1 - np.dot(pos1, pos2) + 1e-7
    omc2 = (1 - np.cos(xi)) / 2
    ret = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
    return np.where(xi == 0, 1, ret)


def vec_orf(pos1, pos2):
    return np.dot(pos1, pos2)


def hd_powerlaw(f, pos1, pos2, log10_A=-15, gamma=4.3):
    return utils.powerlaw(f, log10_A, gamma) * hd_orf(pos1, pos2)


def vec_powerlaw(f, pos1, pos2, log10_A=-15, gamma=4.3):
    return utils.powerlaw(f, log10_A, gamma) * vec_orf(pos1, pos2)


class TestPTASignals(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psrs = [Pulsar(datadir + '/B1855+09_NANOGrav_11yv0.gls.par',
                            datadir + '/B1855+09_NANOGrav_11yv0.tim'),
                     Pulsar(datadir + '/J1909-3744_NANOGrav_11yv0.gls.par',
                            datadir + '/J1909-3744_NANOGrav_11yv0.tim')]

    def test_pta_phi(self):
        T1, T2, T3 = 3.16e8, 3.16e8, 3.16e8
        nf1, nf2, nf3 = 2, 2, 1

        pl = signal_base.Function(utils.powerlaw,
                                  log10_A=parameter.Uniform(-18,-12),
                                  gamma=parameter.Uniform(1,7))
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nf1, Tspan=T1)

        hpl = signal_base.Function(hd_powerlaw,
                                   log10_A=parameter.Uniform(-18,-12),
                                   gamma=parameter.Uniform(1,7))
        crn = gp_signals.FourierBasisCommonGP(crossspectrum=hpl, components=1,
                                              name='gw', Tspan=T3)

        model = rn + crn
        pta = model(self.psrs[0]) + model(self.psrs[1])

        lA1, gamma1 = -13, 1e-15
        lA2, gamma2 = -13.3, 1e-15
        lAc, gammac = -13.1, 1e-15

        params = {'gw_log10_A': lAc, 'gw_gamma': gammac,
                  'B1855+09_log10_A': lA1, 'B1855+09_gamma': gamma1,
                  'J1909-3744_log10_A': lA2, 'J1909-3744_gamma': gamma2}

        phi = pta.get_phi(params)
        phiinv = pta.get_phiinv(params)

        T1, T2, T3 = 3.16e8, 3.16e8, 3.16e8
        nf1, nf2, nf3 = 2, 2, 1

        F1, f1, _ = utils.createfourierdesignmatrix_red(
            self.psrs[0].toas, nf1, freq=True, Tspan=T1)
        F2, f2, _ = utils.createfourierdesignmatrix_red(
            self.psrs[1].toas, nf2, freq=True, Tspan=T2)
        F1c, fc, _ = utils.createfourierdesignmatrix_red(
            self.psrs[0].toas, nf3, freq=True, Tspan=T3)
        F2c, fc, _ = utils.createfourierdesignmatrix_red(
            self.psrs[1].toas, nf3, freq=True, Tspan=T3)

        nftot = 2 * 2 * nf1
        phidiag = np.zeros(nftot)
        phit = np.zeros((nftot, nftot))

        phidiag[:4] = utils.powerlaw(f1, lA1, gamma1) * f1[0]
        phidiag[:2] += utils.powerlaw(fc, lAc, gammac) * fc[0]
        phidiag[4:] = utils.powerlaw(f2, lA2, gamma2) * f2[0]
        phidiag[4:6] += utils.powerlaw(fc, lAc, gammac) * fc[0]

        phit[np.diag_indices(nftot)] = phidiag

        phit[:2, 4:6] = np.diag(hd_powerlaw(fc, self.psrs[0].pos,
                                            self.psrs[1].pos, lAc,
                                            gammac) * fc[0])
        phit[4:6, :2] = np.diag(hd_powerlaw(fc, self.psrs[0].pos,
                                            self.psrs[1].pos, lAc,
                                            gammac) * fc[0])

        msg = 'PTA Phi is incorrect.'
        assert np.allclose(phi, phit, rtol=1e-15, atol=1e-17), msg
        msg = 'PTA Phi inverse is incorrect.'
        assert np.allclose(phiinv, np.linalg.inv(phit),
                           rtol=1e-15, atol=1e-17), msg
