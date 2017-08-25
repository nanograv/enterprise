#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pulsar
----------------------------------

Tests for `pulsar` module. Will eventually want to add tests
for time slicing, PINT integration and pickling.
"""


import unittest
from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle



class TestPulsar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim')

    def test_residuals(self):
        """Check Residual shape."""

        msg = 'Residuals shape incorrect'
        assert self.psr.residuals.shape == (4005,), msg

    def test_toaerrs(self):
        """Check TOA errors shape."""

        msg = 'TOA errors shape incorrect'
        assert self.psr.toaerrs.shape == (4005,), msg

    def test_toas(self):
        """Check TOA shape."""

        msg = 'TOA shape incorrect'
        assert self.psr.toas.shape == (4005,), msg

    def test_freqs(self):
        """Check frequencies shape."""

        msg = 'Frequencies shape incorrect'
        assert self.psr.freqs.shape == (4005,), msg

    def test_flags(self):
        """Check flags shape."""

        msg = 'Flags shape incorrect'
        assert self.psr.flags['f'].shape == (4005,), msg

    def test_backend_flags(self):
        """Check backend_flags shape."""

        msg = 'Backend Flags shape incorrect'
        assert self.psr.backend_flags.shape == (4005,), msg

    def test_sky(self):
        """Check Sky location."""

        sky = (1.4023093811712661, 4.9533700839400492)

        msg = 'Incorrect sky location'
        assert np.allclose(self.psr.theta, sky[0]), msg
        assert np.allclose(self.psr.phi, sky[1]), msg

    def test_design_matrix(self):
        """Check design matrix shape."""

        msg = 'Design matrix shape incorrect.'
        assert self.psr.Mmat.shape == (4005, 91), msg

    def test_filter_data(self):
        """Place holder for filter_data tests."""
        assert self.psr.filter_data() is None

    def test_to_pickle(self):
        """Place holder for to_pickle tests."""
        self.psr.to_pickle()
        with open('B1855+09.pkl', 'rb') as f:
            pkl_psr = pickle.load(f)

        assert np.allclose(self.psr.residuals, pkl_psr.residuals, rtol=1e-10)

        self.psr.to_pickle('pickle_dir')
        with open('pickle_dir/B1855+09.pkl', 'rb') as f:
            pkl_psr = pickle.load(f)

        assert np.allclose(self.psr.residuals, pkl_psr.residuals, rtol=1e-10)

    def test_wrong_input(self):
        """Test exception when incorrect par(tim) file given."""

        with self.assertRaises(IOError) as context:
            Pulsar('wrong.par', 'wrong.tim')

            msg = 'Cannot find parfile wrong.par or timfile wrong.tim!'
            self.assertTrue(msg in context.exception)


class TestPulsarPint(TestPulsar):

    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                         datadir + '/B1855+09_NANOGrav_9yv1.tim',
                         ephem='DE430', timing_package='pint')
