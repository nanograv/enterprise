#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `utils` module.
"""

import unittest

import numpy as np

import enterprise.constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import anis_coefficients as anis
from enterprise.signals import utils
from tests.enterprise_test_data import datadir

import ephem


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.gls.par", datadir + "/B1855+09_NANOGrav_9yv1.tim")

        cls.F, _ = utils.createfourierdesignmatrix_red(cls.psr.toas, nmodes=30)

        cls.Fdm, _ = utils.createfourierdesignmatrix_dm(cls.psr.toas, freqs=cls.psr.freqs, nmodes=30)

        cls.Feph, cls.feph = utils.createfourierdesignmatrix_ephem(cls.psr.toas, cls.psr.pos, nmodes=30)

        cls.Mm = utils.create_stabletimingdesignmatrix(cls.psr.Mmat)

    def test_createstabletimingdesignmatrix(self):
        """Timing model design matrix shape."""

        msg = "Timing model design matrix shape incorrect"
        assert self.Mm.shape == self.psr.Mmat.shape, msg

    def test_createfourierdesignmatrix_red(self, nf=30):
        """Check Fourier design matrix shape."""

        msg = "Fourier design matrix shape incorrect"
        assert self.F.shape == (4005, 2 * nf), msg

    def test_createfourierdesignmatrix_dm(self, nf=30):
        """Check DM-variation Fourier design matrix shape."""

        msg = "DM-variation Fourier design matrix shape incorrect"
        assert self.Fdm.shape == (4005, 2 * nf), msg

    def test_createfourierdesignmatrix_ephem(self, nf=30):
        """Check x-axis ephemeris Fourier design matrix shape."""

        F1, F1f = self.Feph, self.feph

        msg = "Ephemeris Fourier design matrix shape incorrect"
        assert F1.shape == (4005, 6 * nf), msg

        msg = "Ephemeris Fourier design matrix values incorrect"
        assert np.allclose(
            F1[:, 0::3] ** 2 + F1[:, 1::3] ** 2 + F1[:, 2::3] ** 2, (F1[:, 0::3] / self.psr.pos[0]) ** 2
        ), msg

        msg = "Ephemeris frequencies vector shape incorrect"
        assert F1f.shape == (6 * nf,), msg

        msg = "Ephemeris frequencies vector values incorrect"
        assert np.all(F1f[::6] == F1f[5::6]), msg
        assert np.allclose(np.diff(F1f[:-6:6] - F1f[6::6]), 0), msg

    def test_ecc_cw_waveform(self):
        """Check eccentric wafeform generation."""
        nmax = 100
        mc = 5e8
        dl = 300
        h0 = 1e-14
        F = 2e-8
        e = 0.6
        t = self.psr.toas
        l0 = 0.2
        gamma = 0.4
        gammadot = 0.1
        inc = 1.3
        s = utils.calculate_splus_scross(nmax, mc, dl, h0, F, e, t, l0, gamma, gammadot, inc)

        msg = "Single source waveform shape incorrect"
        assert s[0].shape == (4005,), msg
        assert s[1].shape == (4005,), msg

    def test_fplus_fcross(self):
        """Check fplus, fcross generation."""
        gwtheta = 1.4
        gwphi = 2.7
        fplus, fcross, _ = utils.create_gw_antenna_pattern(self.psr.pos, gwtheta, gwphi)

        msg1 = "Fplus value incorrect"
        msg2 = "Fcross value incorrect"
        assert np.allclose(fplus, 0.161508137208), msg1
        assert np.allclose(fcross, -0.130823200124), msg2

    def test_numerical_ecc_integration(self):
        """Test numerical integration of eccentric GW."""
        F0 = 1e-8
        e0 = 0.3
        gamma0 = 0.4
        phase0 = 1.2
        mc = 1e9
        q = 0.25
        t = self.psr.toas - self.psr.toas.min()
        ind = np.argsort(t)
        s = utils.solve_coupled_ecc_solution(F0, e0, gamma0, phase0, mc, q, t[ind])
        s2 = utils.solve_coupled_constecc_solution(F0, e0, phase0, mc, t[ind])
        msg = "Numerical integration failed"
        assert s.shape == (4005, 4), msg
        assert s2.shape == (4005, 2), msg

    def test_quantization_matrix(self):
        """Test quantization matrix generation."""
        U = utils.create_quantization_matrix(self.psr.toas, dt=1)[0]

        msg1 = "Quantization matrix shape incorrect."
        msg2 = "Quantization matrix contains single TOA epochs."
        assert U.shape == (4005, 235), msg1
        assert all(np.sum(U, axis=0) > 1), msg2

        inds = utils.quant2ind(U, as_slice=False)
        slcs = utils.quant2ind(U, as_slice=True)
        inds_check = [utils.indices_from_slice(slc) for slc in slcs]

        msg3 = "Quantization Matrix slice not equal to quantization indices"
        for ind, ind_c in zip(inds, inds_check):
            assert np.all(ind == ind_c), msg3

    def test_indices_from_slice(self):
        """Test conversion of slices to numpy indices"""
        ind_np = np.array([2, 4, 6, 8])
        ind_np_check = utils.indices_from_slice(ind_np)

        msg1 = "Numpy indices not left as-is by indices_from_slice"
        assert np.all(ind_np == ind_np_check), msg1

        slc = slice(2, 10, 2)
        ind_np_check = utils.indices_from_slice(slc)
        msg2 = "Slice not converted properly by indices_from_slice"
        assert np.all(ind_np == ind_np_check), msg2

    def test_psd(self):
        """Test PSD functions."""
        Tmax = self.psr.toas.max() - self.psr.toas.min()
        f = np.linspace(1 / Tmax, 10 / Tmax, 10)
        log10_A = -15
        gamma = 4.33
        lf0 = -8.5
        kappa = 10 / 3
        beta = 0.5
        pl = (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * f[0]
        hcf = 10**log10_A * (f / const.fyr) ** ((3 - gamma) / 2) / (1 + (10**lf0 / f) ** kappa) ** beta
        pt = hcf**2 / 12 / np.pi**2 / f**3 * f[0]

        msg = "PSD calculation incorrect"
        assert np.allclose(utils.powerlaw(f, log10_A, gamma), pl), msg
        assert np.allclose(utils.turnover(f, log10_A, gamma, lf0, kappa, beta), pt), msg

    def test_orf(self):
        """Test ORF functions."""
        p1 = np.array([0.3, 0.648, 0.7])
        p2 = np.array([0.2, 0.775, -0.6])

        # test auto terms
        #
        hd = utils.hd_orf(p1, p1)
        hd_exp = 1.0
        #
        dp = utils.dipole_orf(p1, p1)
        dp_exp = 1.0 + 1e-5
        #
        mp = utils.monopole_orf(p1, p1)
        mp_exp = 1.0 + 1e-5
        #
        psr_positions = np.array([[1.318116071652818, 2.2142974355881808], [1.1372584174390601, 0.79539883018414359]])
        anis_basis = anis.anis_basis(psr_positions, lmax=1)
        anis_orf = round(utils.anis_orf(p1, p1, [0.0, 1.0, 0.0], anis_basis=anis_basis, psrs_pos=[p1, p2], lmax=1), 3)
        anis_orf_exp = 1.147
        #

        msg = "ORF auto term incorrect for {}"
        keys = ["hd", "dipole", "monopole", "anisotropy"]
        vals = [(hd, hd_exp), (dp, dp_exp), (mp, mp_exp), (anis_orf, anis_orf_exp)]
        for key, val in zip(keys, vals):
            assert val[0] == val[1], msg.format(key)

        # test off diagonal terms
        #
        hd = utils.hd_orf(p1, p2)
        omc2 = (1 - np.dot(p1, p2)) / 2
        hd_exp = 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
        #
        dp = utils.dipole_orf(p1, p2)
        dp_exp = np.dot(p1, p2)
        #
        mp = utils.monopole_orf(p1, p2)
        mp_exp = 1.0
        #
        psr_positions = np.array([[1.318116071652818, 2.2142974355881808], [1.1372584174390601, 0.79539883018414359]])
        anis_basis = anis.anis_basis(psr_positions, lmax=1)
        anis_orf = round(utils.anis_orf(p1, p2, [0.0, 1.0, 0.0], anis_basis=anis_basis, psrs_pos=[p1, p2], lmax=1), 3)
        anis_orf_exp = -0.150
        #

        msg = "ORF cross term incorrect for {}"
        keys = ["hd", "dipole", "monopole", "anisotropy"]
        vals = [(hd, hd_exp), (dp, dp_exp), (mp, mp_exp), (anis_orf, anis_orf_exp)]
        for key, val in zip(keys, vals):
            assert val[0] == val[1], msg.format(key)


class TestAstrometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # TODO: use a different pulsar so it's faster?
        # initialize Pulsar class that uses Equatorial Coordinates
        cls.psr = Pulsar(
            datadir + "/1713.Sep.T2.par", datadir + "/1713.Sep.T2.tim", timing_package="tempo2", drop_t2pulsar=False
        )

        cls.Mmat = cls.psr.t2pulsar.designmatrix(fixunits=False, fixsigns=True, incoffset=True)
        cls.posepoch = cls.psr.t2pulsar["POSEPOCH"].val * 86400.0
        cls.dm, cls.dmp = utils.create_astrometry_timing_model(cls.psr.toas, cls.psr._raj, cls.psr._decj, cls.posepoch)

    def test_ddelay_dastrometry(self):
        """Test the derivatives of the astrometry parameters"""

        incorrect_units = ["PMRA", "PMDEC"]

        for pp, pname in enumerate(self.dmp):
            dmc_self = self.dm[:, pp]

            t2dm = ("Offset",) + self.psr.t2pulsar.pars(which="fit")
            pind = t2dm.index(pname)
            dmc_t2 = self.Mmat[self.psr._isort, pind]
            a = dmc_t2
            b = dmc_self

            if pname not in incorrect_units:
                rel_diff = (np.abs(a) - np.abs(b)) / (np.abs(a) + np.abs(b))

                msg = f"ddelay_d{pname} is not consistent with Tempo2"
                assert np.allclose(rel_diff, 0.0, atol=0.05), msg

            else:
                # Proper motion doesn't have the right units
                conv = np.mean(np.abs(a) / np.abs(b))

                rel_diff = (np.abs(a) - conv * np.abs(b)) / (np.abs(a) + conv * np.abs(b))

                msg = f"ddelay_d{pname} is not consistent with Tempo2"
                assert np.allclose(rel_diff, 0.0, atol=0.05), msg

    def test_get_psrname_from_pos(self):
        """Test the functionality to derive pulsar names"""

        # Pulsar B1855+09 (= J1857+09..)
        decj, raj = (0.16848694562363042, 4.9533700839400492)
        eq = ephem.Equatorial(raj, decj, epoch=ephem.J2000)
        ec = ephem.Ecliptic(eq)
        elong, elat = ec.lon * 180 / np.pi, ec.lat * 180 / np.pi

        msg = "Name from elong/elat not consistent with real pulsar name"
        psrname = utils.get_psrname_from_pos(elong=elong, elat=elat, raj=None, decj=None)
        assert psrname == "J1855+0939", msg

        msg = "Name from raj/decj not consistent with real pulsar name"
        psrname = utils.get_psrname_from_pos(elong=None, elat=None, raj=raj, decj=decj)
        assert psrname == "J1855+0939", msg

        with self.assertRaises(ValueError):
            psrname = utils.get_psrname_from_pos(elong=None, elat=None, raj=None, decj=None)
