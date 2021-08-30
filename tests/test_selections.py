#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pulsar
----------------------------------

Tests for `signals/selections` module.
"""

import unittest
import operator
import functools

import numpy as np

from enterprise.pulsar import Pulsar
import enterprise.signals.selections as selections
from tests.enterprise_test_data import datadir


class TestSelections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1937+21_NANOGrav_9yv1.gls.par", datadir + "/B1937+21_NANOGrav_9yv1.tim")

    def test_selections(self):
        # note: -B flag ('by_band') not currently represented in test data
        for sel in ["cut_half", "by_frontend", "by_backend", "nanograv_backends", "by_telescope", "no_selection"]:

            s = selections.Selection(getattr(selections, sel))(self.psr)

            msg = "Selection mask count is incorrect for {}".format(sel)
            assert sum(sum(mask) for mask in s.masks.values()) == len(self.psr.toas), msg

            msg = "Selection mask coverage incomplete for {}".format(sel)
            assert np.all(functools.reduce(operator.or_, s.masks.values())), msg

            msg = "Selection mask not independent for {}".format(sel)
            assert np.all(sum(mask for mask in s.masks.values()) == 1), msg


class TestSelectionsPint(TestSelections):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        cls.psr = Pulsar(
            datadir + "/B1937+21_NANOGrav_9yv1.gls.par", datadir + "/B1937+21_NANOGrav_9yv1.tim", timing_package="pint"
        )
