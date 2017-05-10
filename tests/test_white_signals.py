#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_white_signals
----------------------------------

Tests for white signal modules.
"""


import unittest
import numpy as np

from enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
import enterprise.signals.white_signals as ws


class TestWhiteSignals(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psr = Pulsar(datadir + '/B1855+09_NANOGrav_11yv0.gls.par',
                          datadir + '/B1855+09_NANOGrav_11yv0.tim')

    def test_efac(self):
        """Test that efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        ef = ws.MeasurementNoise(efac=efac)
        efm = ef(self.psr)

        # parameters
        efac = 1.5
        params = {'B1855+09_efac': efac}

        # correct value
        nvec0 = efac**2 * self.psr.toaerrs**2

        # test
        msg = 'EFAC covariance incorrect.'
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_efac_backend(self):
        """Test that backend-efac signal returns correct covariance."""
        # set up signal and parameters
        efac = parameter.Uniform(0.1, 5)
        selection = Selection(selections.by_backend)
        ef = ws.MeasurementNoise(efac=efac, selection=selection)
        efm = ef(self.psr)

        # parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        params = {'B1855+09_efac_430_ASP': efacs[0],
                  'B1855+09_efac_430_PUPPI': efacs[1],
                  'B1855+09_efac_L-wide_ASP': efacs[2],
                  'B1855+09_efac_L-wide_PUPPI': efacs[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct]**2 * self.psr.toaerrs[ind]**2

        # test
        msg = 'EFAC covariance incorrect.'
        assert np.all(efm.get_ndiag(params) == nvec0), msg

    def test_equad(self):
        """Test that equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        eq = ws.EquadNoise(log10_equad=equad)
        eqm = eq(self.psr)

        # parameters
        equad = -6.4
        params = {'B1855+09_log10_equad': equad}

        # correct value
        nvec0 = 10**(2*equad) * np.ones_like(self.psr.toas)

        # test
        msg = 'EQUAD covariance incorrect.'
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_equad_backend(self):
        """Test that backend-equad signal returns correct covariance."""
        # set up signal and parameters
        equad = parameter.Uniform(-10, -5)
        selection = Selection(selections.by_backend)
        eq = ws.EquadNoise(log10_equad=equad, selection=selection)
        eqm = eq(self.psr)

        # parameters
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_log10_equad_430_ASP': equads[0],
                  'B1855+09_log10_equad_430_PUPPI': equads[1],
                  'B1855+09_log10_equad_L-wide_ASP': equads[2],
                  'B1855+09_log10_equad_L-wide_PUPPI': equads[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = 10**(2*equads[ct]) * np.ones(np.sum(ind))

        # test
        msg = 'EQUAD covariance incorrect.'
        assert np.all(eqm.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad(self):
        """Test that addition of efac and equad signal returns
        correct covariance.
        """
        # set up signals
        efac = parameter.Uniform(0.1, 5)
        ef = ws.MeasurementNoise(efac=efac)
        equad = parameter.Uniform(-10, -5)
        eq = ws.EquadNoise(log10_equad=equad)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efac = 1.5
        equad = -6.4
        params = {'B1855+09_efac': efac,
                  'B1855+09_log10_equad': equad}

        # correct value
        nvec0 = efac**2 * self.psr.toaerrs**2
        nvec0 += 10**(2*equad) * np.ones_like(self.psr.toas)

        # test
        msg = 'EFAC/EQUAD covariance incorrect.'
        assert np.all(m.get_ndiag(params) == nvec0), msg

    def test_add_efac_equad_backend(self):
        """Test that addition of efac-backend and equad-backend signal returns
        correct covariance.
        """
        selection = Selection(selections.by_backend)

        efac = parameter.Uniform(0.1, 5)
        equad = parameter.Uniform(-10, -5)
        ef = ws.MeasurementNoise(efac=efac, selection=selection)
        eq = ws.EquadNoise(log10_equad=equad, selection=selection)
        s = ef + eq
        m = s(self.psr)

        # set parameters
        efacs = [1.3, 1.4, 1.5, 1.6]
        equads = [-6.1, -6.2, -6.3, -6.4]
        params = {'B1855+09_efac_430_ASP': efacs[0],
                  'B1855+09_efac_430_PUPPI': efacs[1],
                  'B1855+09_efac_L-wide_ASP': efacs[2],
                  'B1855+09_efac_L-wide_PUPPI': efacs[3],
                  'B1855+09_log10_equad_430_ASP': equads[0],
                  'B1855+09_log10_equad_430_PUPPI': equads[1],
                  'B1855+09_log10_equad_L-wide_ASP': equads[2],
                  'B1855+09_log10_equad_L-wide_PUPPI': equads[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        nvec0 = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            nvec0[ind] = efacs[ct]**2 * self.psr.toaerrs[ind]**2
            nvec0[ind] += 10**(2*equads[ct]) * np.ones(np.sum(ind))

        # test
        msg = 'EFAC/EQUAD covariance incorrect.'
        assert np.all(m.get_ndiag(params) == nvec0), msg
