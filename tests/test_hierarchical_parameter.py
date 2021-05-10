#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_hierarchical_parameter
----------------------------------

Tests for hierarchical parameter functionality
"""


import unittest

import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import parameter, white_signals
from tests.enterprise_test_data import datadir


class TestHierarchicalParameter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

    def test_enterprise_Parameter(self):
        x = parameter.Uniform(0, 1)

        assert issubclass(x, parameter.Parameter)
        self.assertRaises(TypeError, x.get_logpdf, 0.5)

        x1 = x("x1")

        repr_A = "x1:Uniform(pmin=0, pmax=1)"
        repr_B = "x1:Uniform(pmax=1, pmin=0)"
        assert isinstance(x1, parameter.Parameter)
        assert str(x1) == repr_A or str(x1) == repr_B
        assert x1.get_logpdf(0.5) == 0

    def test_enterprise_Function(self):
        def add(a, x=1, y=2):
            return a + x + y

        x = parameter.Uniform(0, 1)
        f = parameter.Function(add, x=x)

        assert issubclass(f, parameter.FunctionBase)

        f1 = f("f1")

        repr_A = "f1(f1_x:Uniform(pmin=0, pmax=1))"
        repr_B = "f1(f1_x:Uniform(pmax=1, pmin=0))"
        assert isinstance(f1, parameter.FunctionBase)
        assert str(f1) == repr_A or str(f1) == repr_B
        assert len(f1.params) == 1
        assert f1(2) == 5
        assert f1(2, 0.5, 7) == 9.5
        assert f1(2, x=0.5, y=7) == 9.5
        assert f1(3, params={"f1_x": 4}) == 9

    def test_Function_of_Function(self):
        def doub(x):
            return 2 * x

        f = parameter.Function(doub, x=parameter.Uniform(0, 1))

        def mult(a, w=2, z=1):
            return a * w * z

        g = parameter.Function(mult, w=parameter.Uniform(2, 3), z=f)

        g1 = g("g1")

        repr_A = "g1_w:Uniform(pmin=2, pmax=3)"
        repr_B = "g1_w:Uniform(pmax=3, pmin=2)"
        assert isinstance(g1, parameter.FunctionBase)

        assert sorted(map(str, g1.params))[0] == repr_A or sorted(map(str, g1.params))[0] == repr_B

        assert g1(2, z=3) == 12
        assert g1(2, w=10, z=3) == 60

        assert g1(2, params={"g1_z_x": 5, "g1_w": 10}) == 200

    def test_powerlaw(self):
        def powerlaw(f, log10_A=-15):
            return (10 ** log10_A) * f ** 2

        pl = parameter.Function(powerlaw, log10_A=parameter.Uniform(0, 5))

        pl1 = pl("pl1")

        repr_A = "pl1(pl1_log10_A:Uniform(pmin=0, pmax=5))"
        repr_B = "pl1(pl1_log10_A:Uniform(pmax=5, pmin=0))"
        assert str(pl1) == repr_A or str(pl1) == repr_B

        fs = np.array([1, 2, 3])

        assert np.allclose(pl1(fs), np.array([1e-15, 4e-15, 9e-15]))
        assert np.allclose(pl1(fs, log10_A=-16), np.array([1e-16, 4e-16, 9e-16]))
        assert np.allclose(pl1(fs, params={"pl1_log10_A": -17}), np.array([1e-16, 4e-16, 9e-16]))

        def log10(A=10 ** -16):
            return np.log10(A)

        log10f = parameter.Function(log10, A=parameter.Uniform(10 ** -17, 10 ** -14))
        pm = parameter.Function(powerlaw, log10_A=log10f)

        pm1 = pm("pm1")

        repr_A = "pm1(pm1_log10_A_A:Uniform(pmin=1e-17, pmax=1e-14))"
        repr_B = "pm1(pm1_log10_A_A:Uniform(pmax=1e-14, pmin=1e-17))"
        assert str(pm1) == repr_A or str(pm1) == repr_B

        assert np.allclose(pm1(fs, log10_A=-13), np.array([1e-13, 4e-13, 9e-13]))
        assert np.allclose(pm1(fs, params={"pm1_log10_A_A": 10 ** -19}), np.array([1e-19, 4e-19, 9e-19]))

    def test_powerlaw_equad(self):
        def powerlaw(f, log10_A=-15):
            return (10 ** log10_A) * f ** 2

        def log10(A=10 ** -16):
            return np.log10(A)

        fquad = white_signals.EquadNoise(
            log10_equad=parameter.Function(log10, A=parameter.Uniform(10 ** -17, 10 ** -14))
        )

        fquad1 = fquad(self.psr)

        repr_A = "[B1855+09_log10_equad_A:Uniform(pmin=1e-17, pmax=1e-14)]"
        repr_B = "[B1855+09_log10_equad_A:Uniform(pmax=1e-14, pmin=1e-17)]"
        assert str(fquad1.params) == repr_A or str(fquad1.params) == repr_B
        assert np.allclose(
            fquad1.get_ndiag(params={"B1855+09_log10_equad_A": 10 ** -14})[:3], np.array([1e-28, 1e-28, 1e-28])
        )
