#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_deterministic_signals
----------------------------------

Tests for deterministic signal module
"""


import unittest

import numpy as np

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import deterministic_signals, parameter, selections, utils
from enterprise.signals.parameter import function
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


@function
def sine_wave(toas, log10_A=-7, log10_f=-8, phase=0.0):
    return 10 ** log10_A * np.sin(2 * np.pi * toas * 10 ** log10_f + phase)


class TestDeterministicSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_bwm(self):
        """Test BWM waveform."""
        log10_h = parameter.Uniform(-20, -11)("bwm_log10_h")
        cos_gwtheta = parameter.Uniform(-1, 1)("bwm_cos_gwtheta")
        gwphi = parameter.Uniform(0, 2 * np.pi)("bwm_gwphi")
        gwpol = parameter.Uniform(0, np.pi)("bwm_gwpol")
        t0 = parameter.Uniform(53000, 57000)("bwm_t0")
        bwm_wf = utils.bwm_delay(log10_h=log10_h, cos_gwtheta=cos_gwtheta, gwphi=gwphi, gwpol=gwpol, t0=t0)
        bwm = deterministic_signals.Deterministic(bwm_wf)
        m = bwm(self.psr)

        # true parameters
        log10_h = -14
        cos_gwtheta = 0.5
        gwphi = 0.5
        gwpol = 0.0
        t0 = 55000
        params = {
            "bwm_log10_h": log10_h,
            "bwm_cos_gwtheta": cos_gwtheta,
            "bwm_gwphi": gwphi,
            "bwm_gwpol": gwpol,
            "bwm_t0": t0,
        }

        d1 = utils.bwm_delay(
            self.psr.toas, self.psr.pos, log10_h=log10_h, cos_gwtheta=cos_gwtheta, gwphi=gwphi, gwpol=gwpol, t0=t0
        )

        # test
        msg = "BWM Delay incorrect"
        assert np.all(m.get_delay(params) == d1), msg

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
        params = {"B1855+09_log10_A": log10_A, "B1855+09_log10_f": log10_f}

        # correct value
        delay = sine_wave(self.psr.toas, log10_A=log10_A, log10_f=log10_f)

        # test
        msg = "Delay incorrect"
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
        params = {
            "B1855+09_430_ASP_log10_A": lAs[0],
            "B1855+09_430_PUPPI_log10_A": lAs[1],
            "B1855+09_L-wide_ASP_log10_A": lAs[2],
            "B1855+09_L-wide_PUPPI_log10_A": lAs[3],
            "B1855+09_430_ASP_log10_f": lfs[0],
            "B1855+09_430_PUPPI_log10_f": lfs[1],
            "B1855+09_L-wide_ASP_log10_f": lfs[2],
            "B1855+09_L-wide_PUPPI_log10_f": lfs[3],
        }

        # correct value
        flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
        delay = np.zeros_like(self.psr.toas)
        for ct, flag in enumerate(np.unique(flags)):
            ind = flag == self.psr.backend_flags
            delay[ind] = sine_wave(self.psr.toas[ind], log10_A=lAs[ct], log10_f=lfs[ct])

        # test
        msg = "Delay incorrect."
        assert np.all(m.get_delay(params) == delay), msg

    def test_physical_ephem_model(self):
        """Test physical ephemeris model"""

        if isinstance(self.psr, enterprise.pulsar.Tempo2Pulsar):
            # define signals with and without epoch TOAs
            eph1 = deterministic_signals.PhysicalEphemerisSignal(sat_orb_elements=True, model="orbel")
            eph2 = deterministic_signals.PhysicalEphemerisSignal(
                sat_orb_elements=True, use_epoch_toas=False, model="orbel"
            )

            # initialize signals
            e1, e2 = eph1(self.psr), eph2(self.psr)

            # set parameters
            params = {
                "d_jupiter_mass": -8.561198198000628e-12,
                "d_neptune_mass": 1.0251757860647059e-11,
                "d_saturn_mass": 6.22114376130324e-12,
                "d_uranus_mass": -2.1157536169469958e-10,
                "frame_drift_rate": 2.874659280396648e-10,
                "jup_orb_elements": np.array(
                    [0.04140015, -0.03422412, 0.01165894, -0.03525219, -0.00406852, 0.0421522]
                ),
                "sat_orb_elements": np.array(
                    [-0.39701798, -0.13322608, -0.05025925, 0.36331171, -0.17080321, 0.25093799]
                ),
            }

            # test against waveform and compare non-epoch and epoch TOA results
            d1 = e1.get_delay(params=params)
            d2 = e2.get_delay(params=params)

            (jup_mjd, jup_orbel, sat_orbel) = utils.get_planet_orbital_elements("orbel")

            d3 = utils.physical_ephem_delay(
                self.psr.toas,
                self.psr.planetssb,
                self.psr.pos_t,
                times=jup_mjd,
                jup_orbit=jup_orbel,
                sat_orbit=sat_orbel,
                **params,
            )

            msg1 = "Signal delay does not match function delay"
            assert np.allclose(d1, d3, rtol=1e-10), msg1
            msg2 = "epoch-TOA delay does not match full TOA delay"
            assert np.allclose(d1, d2, rtol=1e-10), msg2

            # test against pre-computed wafeform
            eph_wf = np.load(datadir + "/phys_ephem_1855_test.npy")
            msg = "Ephemeris delay does not match pre-computed values"
            assert np.allclose(d1, eph_wf, rtol=1e-10), msg

        # test PINT exception raising
        elif isinstance(self.psr, enterprise.pulsar.PintPulsar):
            with self.assertRaises(NotImplementedError) as context:
                eph1 = deterministic_signals.PhysicalEphemerisSignal(sat_orb_elements=True)
                e1 = eph1(self.psr)

                msg = "Physical Ephemeris model is not compatible with PINT "
                msg += "at this time."
                self.assertEqual(msg, str(context.exception))


class TestDeterministicSignalsPint(TestDeterministicSignals):
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
