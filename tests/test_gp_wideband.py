#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_gp_wideband
----------------------------------

Tests for WidebandTimingModel.
"""


import unittest

import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import white_signals, gp_signals, parameter, selections, signal_base
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


class TestWidebandTimingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        cls.psr = Pulsar(
            datadir + "/J1832-0836_NANOGrav_12yv3.wb.gls.par", datadir + "/J1832-0836_NANOGrav_12yv3.wb.tim"
        )

    def test_wideband(self):
        ms = white_signals.MeasurementNoise(selection=Selection(selections.by_backend))

        dm = gp_signals.WidebandTimingModel(
            dmefac=parameter.Uniform(0.9, 1.1),
            dmefac_selection=Selection(selections.by_backend),
            dmjump=parameter.Normal(0, 1),
            dmjump_selection=Selection(selections.by_frontend),
        )

        model = ms + dm

        pta = signal_base.PTA([model(self.psr)])

        ps = parameter.sample(pta.params)

        pta.get_lnlikelihood(ps)

        dmtiming = pta.pulsarmodels[0].signals[1]

        msg = "DMEFAC masks do not cover the data."
        assert np.all(sum(dmtiming._dmefac_masks) == 1), msg

        msg = "DMJUMP masks do not cover the data."
        assert np.all(sum(dmtiming._dmjump_masks) == 1), msg

        # start with zero DMEFAC and DMJUMP
        p0 = {par.name: (1 if "dmefac" in par.name else 0) for par in dmtiming.params}

        pta.get_lnlikelihood(params=p0)

        phi0 = dmtiming.get_phi(params=p0)
        dl0 = dmtiming.get_delay(params=p0)

        dm_flags, dme_flags = np.array(self.psr.flags["pp_dm"], "d"), np.array(self.psr.flags["pp_dme"], "d")

        delays = np.zeros_like(self.psr.toas)

        check = 0
        for index, par in enumerate(self.psr.fitpars):
            if "DMX" not in par:
                msg = "Problem with unbound timing parameters"
                assert phi0[index] == 1e40, msg
            else:
                dmx = self.psr.dmx[par]
                which = (dmx["DMXR1"] <= (self.psr.stoas / 86400)) & ((self.psr.stoas / 86400) < dmx["DMXR2"])
                check += which

                avgdm = np.sum(dm_flags[which] / dme_flags[which] ** 2) / np.sum(1.0 / dme_flags[which] ** 2)
                vardm = 1.0 / np.sum(1.0 / dme_flags[which] ** 2)

                msg = "Priors do not match"
                assert np.allclose(vardm, phi0[index]), msg

                delays[which] = (avgdm - self.psr.dm - dmx["DMX"]) / (2.41e-4 * self.psr.freqs[which] ** 2)

        msg = "Not all TOAs are covered by DMX"
        assert np.all(check == 1)

        msg = "Delays do not match"
        assert np.allclose(dl0, delays), msg

        # sample DMEFACs randomly
        p1 = {par.name: (parameter.sample(par)[par.name] if "dmefac" in par.name else 0) for par in dmtiming.params}

        pta.get_lnlikelihood(params=p1)

        phi1 = dmtiming.get_phi(params=p1)
        dl1 = dmtiming.get_delay(params=p1)

        sel = Selection(selections.by_backend)(self.psr)
        msg = "Problem making selection"
        assert np.all(sum(m for m in sel.masks.values()) == 1), msg

        dme_flags_var = dme_flags.copy()

        for key, mask in sel.masks.items():
            dmefac = p1["J1832-0836_" + key + "_dmefac"]
            dme_flags_var[mask] *= dmefac

        for index, par in enumerate(self.psr.fitpars):
            if "DMX" not in par:
                msg = "Problem with unbound timing parameters"
                assert phi1[index] == 1e40, msg
            else:
                dmx = self.psr.dmx[par]
                which = (dmx["DMXR1"] <= (self.psr.stoas / 86400)) & ((self.psr.stoas / 86400) < dmx["DMXR2"])

                avgdm = np.sum(dm_flags[which] / dme_flags_var[which] ** 2) / np.sum(1.0 / dme_flags_var[which] ** 2)
                vardm = 1.0 / np.sum(1.0 / dme_flags_var[which] ** 2)

                msg = "Priors do not match"
                assert np.allclose(vardm, phi1[index]), msg

                delays[which] = (avgdm - self.psr.dm - dmx["DMX"]) / (2.41e-4 * self.psr.freqs[which] ** 2)

        msg = "Delays do not match"
        assert np.allclose(dl1, delays), msg


class TestGPSignalsPint(TestWidebandTimingModel):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        cls.psr = Pulsar(
            datadir + "/J1832-0836_NANOGrav_12yv3.wb.gls.par",
            datadir + "/J1832-0836_NANOGrav_12yv3.wb.tim",
            timing_package="pint",
        )

    def test_wideband(self):
        # PINT Pulsar object needs to include dm information

        pass
