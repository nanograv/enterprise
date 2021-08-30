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

from enterprise.pulsar import Pulsar
from tests.enterprise_test_data import datadir


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
        assert self.psr.dm == np.longdouble("13.299393"), msg

        msg = "dmx struct incorrect (spotcheck)"
        assert len(self.psr.dmx) == 72, msg
        assert self.psr.dmx["DMX_0001"]["DMX"] == np.longdouble("0.015161863"), msg
        assert self.psr.dmx["DMX_0001"]["fit"], msg

    def test_freqs(self):
        """Check frequencies shape."""

        msg = "Frequencies shape incorrect"
        assert self.psr.freqs.shape == (4005,), msg

    def test_flags(self):
        """Check flags shape."""

        msg = "Flags shape incorrect"
        assert self.psr.flags["f"].shape == (4005,), msg

    def test_backend_flags(self):
        """Check backend_flags shape."""

        msg = "Backend Flags shape incorrect"
        assert self.psr.backend_flags.shape == (4005,), msg

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
    def test_deflate_inflate(self):
        psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

        dm = psr._designmatrix.copy()

        psr.deflate()
        psr.to_pickle()

        with open("B1855+09.pkl", "rb") as f:
            pkl_psr = pickle.load(f)
        pkl_psr.inflate()

        assert np.allclose(dm, pkl_psr._designmatrix)

        del pkl_psr

        psr.destroy()

        with open("B1855+09.pkl", "rb") as f:
            pkl_psr = pickle.load(f)

        with self.assertRaises(FileNotFoundError):
            pkl_psr.inflate()

    def test_wrong_input(self):
        """Test exception when incorrect par(tim) file given."""

        with self.assertRaises(IOError) as context:
            Pulsar("wrong.par", "wrong.tim")

            msg = "Cannot find parfile wrong.par or timfile wrong.tim!"
            self.assertTrue(msg in context.exception)

    def test_value_error(self):
        """Test exception when unknown argument is given"""

        with self.assertRaises(ValueError):
            Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.time")


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

    # exclude tests pending implementation of .stoas, .dm, .dmx in PintPulsar

    def test_stoas(self):
        assert hasattr(self.psr, "stoas")

    def test_dm(self):
        assert hasattr(self.psr, "dm")

    def test_planetssb(self):
        assert hasattr(self.psr, "planetssb")

    def test_sunssb(self):
        assert hasattr(self.psr, "sunssb")

    def test_model(self):
        assert hasattr(self.psr, "model")

    def test_pint_toas(self):
        assert hasattr(self.psr, "pint_toas")

    def test_deflate_inflate(self):
        pass
