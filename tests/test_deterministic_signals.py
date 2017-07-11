#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_deterministic_signals
----------------------------------

Tests for deterministic signal module
"""


import unittest
import numpy as np

from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base


@signal_base.function
def sine_wave(toas, log10_A=-7, log10_f=-8, phase=0.0):
    return 10**log10_A * np.sin(2*np.pi*toas*10**log10_f + phase)


class TestDeterministicSignals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim')

    def test_delay(self):
        """Test deterministic signal no selection."""
        # set up signal and parameters
        log10_Ad = parameter.Uniform(-10, -5)
        log10_fd = parameter.Uniform(-9, -7)
        waveform = sine_wave(log10_A=log10_Ad, log10_f=log10_fd)
        dt = deterministic_signals.Deterministic(waveform)
        m = dt(self.psr)

        # parameters
        log10_A = -7.2
        log10_f = -8.0
        params = {'B1855+09_log10_A': log10_A,
                  'B1855+09_log10_f':log10_f}

        # correct value
        delay = sine_wave(self.psr.toas, log10_A=log10_A, log10_f=log10_f)

        # test
        msg = 'Delay incorrect'
        assert np.all(m.get_delay(params) == delay), msg

    def test_delay_backend(self):
        """Test deterministic signal with selection."""
        # set up signal and parameters
        log10_Ad = parameter.Uniform(-10, -5)
        log10_fd = parameter.Uniform(-9, -7)
        waveform = sine_wave(log10_A=log10_Ad, log10_f=log10_fd)
        selection = Selection(selections.by_backend)
        dt = deterministic_signals.Deterministic(waveform, selection=selection)
        m = dt(self.psr)

        # parameters
        lAs = [-7.6, -7.1, -6, -6.4]
        lfs = [-7.6, -8.0, -9, -8.4]
        params = {'B1855+09_430_ASP_log10_A': lAs[0],
                  'B1855+09_430_PUPPI_log10_A': lAs[1],
                  'B1855+09_L-wide_ASP_log10_A': lAs[2],
                  'B1855+09_L-wide_PUPPI_log10_A': lAs[3],
                  'B1855+09_430_ASP_log10_f': lfs[0],
                  'B1855+09_430_PUPPI_log10_f': lfs[1],
                  'B1855+09_L-wide_ASP_log10_f': lfs[2],
                  'B1855+09_L-wide_PUPPI_log10_f': lfs[3]}

        # correct value
        flags = ['430_ASP', '430_PUPPI', 'L-wide_ASP', 'L-wide_PUPPI']
        delay = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            delay[ind] = sine_wave(self.psr.toas[ind], log10_A=lAs[ct],
                                   log10_f=lfs[ct])

        # test
        msg = 'Delay incorrect.'
        assert np.all(m.get_delay(params) == delay), msg


class TestDeterministicSignalsPint(TestDeterministicSignals):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim',
                         ephem='DE430', timing_package='pint')
