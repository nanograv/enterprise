#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pulsar
----------------------------------

Tests for `pulsar` module. Will eventually want to add tests
for time slicing, PINT integration and pickling.
"""

import sys
import os
import shutil
import unittest
import pickle
import pytest

import numpy as np

try:
    import libstempo as t2
except (ImportError, RuntimeError):
    t2 = None

from enterprise.pulsar import Pulsar
from tests.enterprise_test_data import datadir

import pint.models.timing_model
from pint.models import get_model_and_toas


# FIXME: these really should all be run for all available Pulsar types
# But that's an annoying conversion
class TestPulsar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("pickle_dir", ignore_errors=True)

    def test_residuals(self):
        """Check Residual shape."""

        msg = "Residuals shape incorrect"
        assert self.psr.residuals.shape == (4005,), msg

    def test_toaerrs(self):
        """Check TOA errors shape."""

        msg = "TOA errors shape incorrect"
        assert self.psr.toaerrs.shape == (4005,), msg

    def test_toas(self):
        """Check TOA shape."""

        msg = "TOA shape incorrect"
        assert self.psr.toas.shape == (4005,), msg

    def test_stoas(self):
        """Check STOA shape."""

        msg = "stoa shape incorrect"
        assert self.psr.stoas.shape == (4005,), msg

    def test_dm(self):
        """Check DM/DMX access."""

        msg = "dm value incorrect"
        assert self.psr.dm == 13.299393, msg

        msg = "dmx struct incorrect (spotcheck)"
        assert len(self.psr.dmx) == 72, msg
        assert self.psr.dmx["DMX_0001"]["DMX"] == 0.015161863, msg
        assert self.psr.dmx["DMX_0001"]["fit"], msg

    def test_freqs(self):
        """Check frequencies shape."""

        msg = "Frequencies shape incorrect"
        assert self.psr.freqs.shape == (4005,), msg

    def test_flags(self):
        """Check flags shape, content, and setting"""

        msg = "Flags shape incorrect"
        assert self.psr.flags["f"].shape == (4005,), msg

        msg = "Flag content or sorting incorrect"
        assert np.all(self.psr._flags["fe"][self.psr._isort] == self.psr.flags["fe"]), msg

        # only possible if flags are stored as dict
        if isinstance(self.psr._flags, dict):
            self.psr.set_flags("name2", self.psr.flags["name"])

            msg = "Setting flags returns incorrect match"
            assert np.all(self.psr.flags["name"] == self.psr.flags["name2"])

    def test_backend_flags(self):
        """Check backend_flags shape and content"""

        msg = "Backend flags shape incorrect"
        assert self.psr.backend_flags.shape == (4005,), msg

        # for the test pulsar, backend should be the same as 'f'
        msg = "Flag content or sorting incorrect"
        assert np.all(self.psr._flags["f"][self.psr._isort] == self.psr.backend_flags), msg

    def test_sky(self):
        """Check Sky location."""

        sky = (1.4023093811712661, 4.9533700839400492)

        msg = "Incorrect sky location"
        assert np.allclose(self.psr.theta, sky[0]), msg
        assert np.allclose(self.psr.phi, sky[1]), msg

    def test_design_matrix(self):
        """Check design matrix shape."""

        msg = "Design matrix shape incorrect."
        assert self.psr.Mmat.shape == (4005, 91), msg

    def test_filter_data(self):
        """Place holder for filter_data tests."""
        assert self.psr.filter_data() is None

    def test_planetssb(self):
        """Place holder for filter_data tests."""
        assert hasattr(self.psr, "planetssb")

    def test_sunssb(self):
        """Place holder for filter_data tests."""
        assert hasattr(self.psr, "sunssb")

    def test_to_pickle(self):
        """Place holder for to_pickle tests."""
        self.psr.to_pickle()
        with open("B1855+09.pkl", "rb") as f:
            pkl_psr = pickle.load(f)

        os.remove("B1855+09.pkl")

        assert np.allclose(self.psr.residuals, pkl_psr.residuals, rtol=1e-10)

        self.psr.to_pickle("pickle_dir")
        with open("pickle_dir/B1855+09.pkl", "rb") as f:
            pkl_psr = pickle.load(f)

        assert np.allclose(self.psr.residuals, pkl_psr.residuals, rtol=1e-10)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python >= 3.8")
@pytest.mark.parametrize(
    "timing_package",
    [
        pytest.param("tempo2", marks=pytest.mark.skipif(t2 is None, reason="TEMPO2 not available")),
        pytest.param("pint", marks=pytest.mark.xfail(reason="FIXME: PintPulsar doesn't do deflate/inflate yet")),
    ],
)
def test_deflate_inflate(timing_package):
    psr = Pulsar(
        datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
        datadir + "/B1855+09_NANOGrav_9yv1.tim",
        timing_package=timing_package,
    )

    dm = psr._designmatrix.copy()

    psr.deflate()
    psr.to_pickle()

    with open("B1855+09.pkl", "rb") as f:
        pkl_psr = pickle.load(f)
    pkl_psr.inflate()

    assert np.array_equal(dm, pkl_psr._designmatrix)

    del pkl_psr

    psr.destroy()

    with open("B1855+09.pkl", "rb") as f:
        pkl_psr = pickle.load(f)

    with pytest.raises(FileNotFoundError):
        pkl_psr.inflate()


@pytest.mark.parametrize(
    "timing_package",
    [
        pytest.param("tempo2", marks=pytest.mark.skipif(t2 is None, reason="TEMPO2 not available")),
        "pint",
    ],
)
def test_wrong_input(timing_package):
    """Test exception when incorrect par(tim) file given."""

    with pytest.raises(IOError) as context:
        Pulsar("wrong.par", "wrong.tim", timing_package=timing_package)

        msg = "Cannot find parfile wrong.par or timfile wrong.tim!"
        assert msg in context.exception


@pytest.mark.parametrize(
    "timing_package",
    [
        pytest.param("tempo2", marks=pytest.mark.skipif(t2 is None, reason="TEMPO2 not available")),
        "pint",
    ],
)
def test_value_error(timing_package):
    """Test exception when unknown argument is given"""

    with pytest.raises(ValueError):
        Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.time",
            timing_package=timing_package,
        )


class TestPulsarPint(TestPulsar):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            drop_pintpsr=False,
            timing_package="pint",
        )

    def test_deflate_inflate(self):
        pass

    def test_load_radec_psr(self):
        """Setup the Pulsar object."""

        with self.assertRaises(pint.models.timing_model.TimingModelError):
            # initialize Pulsar class with RAJ DECJ and PMLAMBDA, PMBETA
            Pulsar(
                datadir + "/J0030+0451_RADEC_wrong.par",
                datadir + "/J0030+0451_NANOGrav_9yv1.tim",
                ephem="DE430",
                drop_pintpsr=False,
                timing_package="pint",
            )

    def test_no_planet(self):
        """Test exception when incorrect par(tim) file given."""

        with self.assertRaises(ValueError) as context:
            model, toas = get_model_and_toas(
                datadir + "/J0030+0451_NANOGrav_9yv1.gls.par", datadir + "/J0030+0451_NANOGrav_9yv1.tim", planets=False
            )
            Pulsar(model, toas, planets=True)
            msg = "obs_earth_pos is not in toas.table.colnames. Either "
            msg += "`planet` flag is not True in `toas` or further Pint "
            msg += "development to add additional planets is needed."
            self.assertTrue(msg in context.exception)
