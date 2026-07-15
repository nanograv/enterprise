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

from enterprise.pulsar import Pulsar, MockPulsar
from tests.enterprise_test_data import datadir
from tests.enterprise_test_data import LIBSTEMPO_INSTALLED, PINT_INSTALLED

import ephem

if PINT_INSTALLED:
    import pint.models.timing_model
    from pint.models import get_model_and_toas


@pytest.mark.skipif(not LIBSTEMPO_INSTALLED, reason="Skipping tests that require libstempo because it isn't installed")
class TestTimingPackageExceptions(unittest.TestCase):
    def test_unkown_timing_package(self):
        # initialize Pulsar class
        with self.assertRaises(ValueError):
            self.psr = Pulsar(
                datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
                datadir + "/B1855+09_NANOGrav_9yv1.tim",
                timing_package="foobar",
            )

    def test_clk_but_no_bipm(self):
        self.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            clk="TT(BIPM2020)",
            timing_package="pint",
        )


@pytest.mark.skipif(not LIBSTEMPO_INSTALLED, reason="Skipping tests that require libstempo because it isn't installed")
class TestPulsar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim", drop_t2pulsar=True
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("pickle_dir", ignore_errors=True)

    def test_droppsr(self):
        self.psr_nodrop = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim", drop_t2pulsar=False
        )

        self.psr_nodrop.drop_tempopsr()

        with self.assertRaises(AttributeError):
            _ = self.psr.t2pulsar

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

        os.remove("B1855+09.pkl")

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

    def test_to_feather(self):
        """Test creating feather file from Pulsar method"""

        self.psr.to_feather("test.feather")
        assert os.path.exists("test.feather")

        loaded_psr = Pulsar("test.feather")
        assert np.allclose(self.psr.residuals, loaded_psr.residuals, rtol=1e-10)

        os.remove("test.feather")


@pytest.mark.skipif(not PINT_INSTALLED, reason="Skipping tests that require PINT because it isn't installed")
class TestPulsarPint(TestPulsar):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            drop_pintpsr=True,
            timing_package="pint",
        )

    def test_droppsr(self):
        self.psr_nodrop = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            drop_pintpsr=False,
            timing_package="pint",
        )

        self.psr_nodrop.drop_pintpsr()

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.model

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.parfile

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.pint_toas

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.timfile

    def test_drop_not_picklable(self):
        self.psr_nodrop = Pulsar(
            datadir + "/B1855+09_NANOGrav_9yv1.gls.par",
            datadir + "/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            drop_pintpsr=False,
            timing_package="pint",
        )

        self.psr_nodrop.drop_not_picklable()

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.model

        with self.assertRaises(AttributeError):
            _ = self.psr_nodrop.pint_toas

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

    def test_load_radec_psr_mdc(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class with RAJ DECJ so _get_radec can be covered
        psr = Pulsar(
            datadir + "/mdc1/J0030+0451.par",
            datadir + "/mdc1/J0030+0451.tim",
            ephem="DE430",
            drop_pintpsr=False,
            timing_package="pint",
        )

        msg = f"Pulsar not loaded properly {self.psr.Mmat.shape}"
        assert psr.Mmat.shape == (130, 8), msg

    def test_no_planet(self):
        """Test exception when incorrect par(tim) file given."""

        with self.assertRaises(ValueError) as context:
            model, toas = get_model_and_toas(
                datadir + "/J0030+0451_NANOGrav_9yv1.gls.par", datadir + "/J0030+0451_NANOGrav_9yv1.tim", planets=False
            )
            Pulsar(model, toas, planets=True, drop_pintpsr=False)
            msg = "obs_earth_pos is not in toas.table.colnames. Either "
            msg += "`planet` flag is not True in `toas` or further Pint "
            msg += "development to add additional planets is needed."
            self.assertTrue(msg in context.exception)


class TestPulsarMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup MockPulsar objects."""
        cls.ntoa = 4005
        cls.toas = np.linspace(53000.0, 58000.0, cls.ntoa)
        cls.flags = {"f": np.array(["nosystem"] * cls.ntoa), "fe": np.array(["nofrontend"] * cls.ntoa)}
        cls.decj, cls.raj = (0.16848694562363042, 4.9533700839400492)
        eq = ephem.Equatorial(cls.raj, cls.decj, epoch=ephem.J2000)
        ec = ephem.Ecliptic(eq)
        cls.elong, cls.elat = ec.lon * 180 / np.pi, ec.lat * 180 / np.pi
        cls.name = "J1855+0939"

        cls.psr = MockPulsar(
            obs_times_mjd=cls.toas,
            name=cls.name,
            elong=cls.elong,
            elat=cls.elat,
            freqs_mhz=1440.0,
            residuals=np.zeros_like(cls.toas),
            toaerrs=1e-6,
            sort=True,
            flags=cls.flags,
            telescope="GBT",
            spindown_order=2,
            inc_astrometry=True,
            posepoch_mjd=55500.0,
            pepoch_mjd=55500.0,
            dm=13.3,
            distance_kpc=1.2,
        )

        cls.psr_spin = MockPulsar(
            obs_times_mjd=cls.toas,
            name=cls.name,
            raj=cls.raj,
            decj=cls.decj,
            freqs_mhz=1440.0 * np.ones_like(cls.toas),
            residuals=np.zeros_like(cls.toas),
            toaerrs=1e-6 * np.ones_like(cls.toas),
            sort=True,
            flags=cls.flags,
            telescope="GBT",
            spindown_order=2,
            inc_astrometry=False,
        )
        cls.psr_spin.set_residuals(np.ones_like(cls.toas))

    def test_design_matrix(self):
        assert self.psr.Mmat.shape == (self.ntoa, 8)
        assert self.psr.fitpars == ["Offset", "Poly1", "Poly2", "RAJ", "DECJ", "PMRA", "PMDEC", "PX"]
        assert self.psr_spin.Mmat.shape == (self.ntoa, 3)
        assert self.psr_spin.fitpars == ["Offset", "Poly1", "Poly2"]
        assert self.psr.setpars == []

    def test_interface_attributes(self):
        assert self.psr.name == self.name
        assert self.psr.telescope.shape == (self.ntoa,)
        assert np.all(self.psr.telescope == "GBT")
        assert self.psr.dm == 13.3
        assert self.psr.dmx == {}
        assert self.psr.planetssb.shape == (self.ntoa, 9, 6)
        assert self.psr.sunssb.shape == (self.ntoa, 6)
        assert self.psr.pos_t.shape == (self.ntoa, 3)
        assert self.psr.pdist == (1.2, 0.24)
        assert np.allclose(self.psr.freqs, 1440.0)
        assert set(self.psr.flags.keys()) == {"f", "fe"}
        assert len(self.psr.backend_flags) == self.ntoa

    def test_set_residuals(self):
        assert np.all(self.psr_spin.residuals == np.ones(self.ntoa))

    def test_requires_name_and_position(self):
        with self.assertRaises(ValueError):
            MockPulsar(obs_times_mjd=self.toas[:10], name="", raj=self.raj, decj=self.decj)
        with self.assertRaises(ValueError):
            MockPulsar(obs_times_mjd=self.toas[:10], name="J0000+0000")

    def test_filter_data(self):
        psr = MockPulsar(
            obs_times_mjd=self.toas,
            name=self.name,
            raj=self.raj,
            decj=self.decj,
            freqs_mhz=1440.0,
            flags=self.flags,
            telescope="GBT",
            inc_astrometry=False,
        )
        psr.filter_data(start_time=54000.0, end_time=56000.0)
        assert len(psr.toas) < self.ntoa
        assert psr.Mmat.shape[0] == len(psr.toas)
        assert psr.telescope.shape == (len(psr.toas),)
        assert psr.planetssb.shape[0] == len(psr.toas)
        assert psr.sunssb.shape[0] == len(psr.toas)
        assert psr.pos_t.shape[0] == len(psr.toas)

    def test_to_pickle(self):
        outdir = "test_mock_pickle_tmp"
        try:
            self.psr.to_pickle(outdir)
            path = os.path.join(outdir, self.name + ".pkl")
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            assert loaded.name == self.psr.name
            assert np.allclose(loaded.residuals, self.psr.residuals)
            assert loaded.Mmat.shape == self.psr.Mmat.shape
        finally:
            if os.path.exists(outdir):
                shutil.rmtree(outdir)

    def test_timing_model_and_red_noise(self):
        """MockPulsar works with TimingModel + Fourier red noise likelihood."""
        from enterprise.signals import parameter, white_signals, gp_signals, utils
        from enterprise.signals.signal_base import PTA

        efac = parameter.Constant(1.0)
        ef = white_signals.MeasurementNoise(efac=efac)
        tm = gp_signals.TimingModel()
        log10_A = parameter.Constant(-15.0)
        gamma = parameter.Constant(13.0 / 3.0)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl, components=10)
        model = ef + tm + rn
        pta = PTA([model(self.psr_spin)])
        lnlike = pta.get_lnlikelihood({})
        assert np.isfinite(lnlike)

    def test_auto_name_helper(self):
        from enterprise.signals import utils as ent_utils

        auto = ent_utils.get_psrname_from_pos(raj=self.raj, decj=self.decj)
        assert auto.startswith("J")


class TestDuckTyping(unittest.TestCase):
    """Test the duck-typing interface detection functions."""

    def test_duck_typing_functions(self):
        """Test the duck-typing interface detection functions."""
        from enterprise.pulsar import _has_pint_toas_interface, _has_pint_model_interface, _has_tempo2_interface

        # Test with objects that have none of the required attributes
        class EmptyObj:
            pass

        empty = EmptyObj()
        assert not _has_pint_toas_interface(empty)
        assert not _has_pint_model_interface(empty)
        assert not _has_tempo2_interface(empty)

        # Test with objects that have some but not all required attributes
        class PartialPintToas:
            def get_mjds(self):
                pass

            def get_errors(self):
                pass

            # Missing get_flags, get_obss, ntoas

        partial_toas = PartialPintToas()
        assert not _has_pint_toas_interface(partial_toas)

        class NonCallablePintToas:
            get_mjds = None
            get_errors = None
            get_flags = None
            get_obss = None
            ntoas = 1

        noncallable_toas = NonCallablePintToas()
        assert not _has_pint_toas_interface(noncallable_toas)

        # Test with objects that have all required attributes
        class MockPintToas:
            def get_mjds(self):
                pass

            def get_errors(self):
                pass

            def get_flags(self):
                pass

            def get_obss(self):
                pass

            @property
            def ntoas(self):
                return 1

        mock_toas = MockPintToas()
        assert _has_pint_toas_interface(mock_toas)

        class MockPintPSR:
            @property
            def value(self):
                return "test"

        class MockPintModel:
            @property
            def PSR(self):
                return MockPintPSR()

            def get_barycentric_toas(self):
                pass

            def designmatrix(self):
                pass

            def barycentric_radio_freq(self):
                pass

            @property
            def params(self):
                return {}

        mock_model = MockPintModel()
        assert _has_pint_model_interface(mock_model)

        class MockPintModelBadPSR:
            @property
            def PSR(self):
                return "test"

            def get_barycentric_toas(self):
                pass

            def designmatrix(self):
                pass

            def barycentric_radio_freq(self):
                pass

            @property
            def params(self):
                return {}

        mock_model_bad_psr = MockPintModelBadPSR()
        assert not _has_pint_model_interface(mock_model_bad_psr)

        class MockTempo2:
            def toas(self):
                return [1, 2, 3]

            @property
            def stoas(self):
                return [1, 2, 3]

            def residuals(self):
                return [1, 2, 3]

            @property
            def toaerrs(self):
                return [1, 2, 3]

            def designmatrix(self):
                pass

            def ssbfreqs(self):
                return [1, 2, 3]

            def telescope(self):
                return ["GBT"]

            def flags(self):
                return {}

            def flagvals(self, key):
                return np.array([], dtype="U1")

            def pars(self, which=None):
                return {}

            @property
            def psrPos(self):
                return np.zeros((3,))

            def __getitem__(self, key):
                return None

            @property
            def name(self):
                return "test"

        mock_tempo2 = MockTempo2()
        assert _has_tempo2_interface(mock_tempo2)

        class MockTempo2BadToas(MockTempo2):
            @property
            def toas(self):
                return [1, 2, 3]

        class MockTempo2BadFlagvals(MockTempo2):
            flagvals = None

        assert not _has_tempo2_interface(MockTempo2BadToas())
        assert not _has_tempo2_interface(MockTempo2BadFlagvals())
