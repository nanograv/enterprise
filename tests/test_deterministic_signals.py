#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deterministic Signals are those that provide a `get_delay()` method,
and are created by the class factories in :mod:`enterprise.signals.deterministic_signals`.
All tests in this module are run on `B1855+09_NANOGrav_9yv1`.
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
    """A simple sine wave Enterprise function object. When instantiated,
    it will create named Parameters for `log10_A`, `log10_f`, `phase`,
    and it will automatically extract `toas` from the linked `Pulsar` object. """

    return 10 ** log10_A * np.sin(2 * np.pi * toas * 10 ** log10_f + phase)


class TestDeterministicSignals(unittest.TestCase):
    """Tests deterministic signals with a tempo2 Pulsar object."""

    @classmethod
    def setUpClass(cls):
        """Set up the :func:`enterprise.Pulsar` object used in tests (tempo2 version)."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_bwm(self):
        """Tests :meth:`enterprise.signals.deterministic_signals.Deterministic`
        using the burst-with-memory function :func:`enterprise.signals.utils.bwm_delay`.
        The test instantiates a deterministic Signal on our test pulsar, and
        compares the array returned by calling `get_delay()` on the Signal
        with a fixed dictionary of parameters, with the result of calling
        the function directly with those parameters.
        """

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
        """Same as :meth:`TestDeterministicSignals.test_bwm`, but
        for a simple sine wave signal."""

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
        """Same as :meth:`TestDeterministicSignals.test_delay`, but
        instantiates the Signal with :func:`enterprise.signals.selections.by_backend`,
        which creates separated named parameters for 430_ASP, 430_PUPPI,
        L-wide_ASP, L-wide_PUPPI. The parameters are automatically accounted for
        in `get_delay()`, but they need to be used explicitly when calling the
        function directly. The tests therefore reconstructs the delay vector by
        building selection masks from :meth:`enterprise.Pulsar.backend_flags`."""

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
        """Tests physical ephemeris model (which is implemented as a deterministic signal)
        four ways:

        - computed directly with :func:`enterprise.signals.utils.physical_ephem_delay`;
        - computed with :meth:`enterprise.signals.deterministic_signals.PhysicalEphemerisSignal.get_delay`
          with `use_epoch_toas=True` (the default), which reduces computation by evaluating ephemeris corrections
          once per measurement epoch, and then interpolating to the full `toas` vector;
        - computed with :meth:`enterprise.signals.deterministic_signals.PhysicalEphemerisSignal.get_delay`,
          setting `use_epoch_toas=False`;
        - loaded from a golden copy.
        """

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

    def test_physical_ephem_model_setIII(self):
        """Test physical ephemeris model"""

        # define signals with and without epoch TOAs
        eph1 = deterministic_signals.PhysicalEphemerisSignal(sat_orb_elements=True, model="setIII")
        eph2 = deterministic_signals.PhysicalEphemerisSignal(
            sat_orb_elements=True, use_epoch_toas=False, model="setIII"
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
            "jup_orb_elements": np.array([0.04140015, -0.03422412, 0.01165894, -0.03525219, -0.00406852, 0.0421522]),
            "sat_orb_elements": np.array([-0.39701798, -0.13322608, -0.05025925, 0.36331171, -0.17080321, 0.25093799]),
        }

        # test against waveform and compare non-epoch and epoch TOA results
        d1 = e1.get_delay(params=params)
        d2 = e2.get_delay(params=params)

        (jup_mjd, jup_orbel, sat_orbel) = utils.get_planet_orbital_elements("setIII")

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


class TestDeterministicSignalsPint(TestDeterministicSignals):
    """Tests deterministic signals with a PINT Pulsar object."""

    @classmethod
    def setUpClass(cls):
        """Set up the :func:`enterprise.Pulsar` object used in tests (PINT version)."""

        # initialize Pulsar class
        cls.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            timing_package="pint",
        )
