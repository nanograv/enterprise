#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `utils` module.
"""

import os
import unittest
import pytest

import numpy as np

import enterprise.constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import anis_coefficients as anis
from enterprise.signals import utils
from tests.enterprise_test_data import LIBSTEMPO_INSTALLED, PINT_INSTALLED
from tests.enterprise_test_data import datadir

import ephem

try:
    import astropy.units as u
    import astropy.constants as ac

    ASTROPY_INSTALLED = True
except ImportError:  # pragma: no cover
    ASTROPY_INSTALLED = False

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        cls.psr = Pulsar(datadir + "/B1855+09_NANOGrav_9yv1.t2.feather")

        cls.F, _ = utils.createfourierdesignmatrix_red(cls.psr.toas, nmodes=30)

        cls.Fdm, _ = utils.createfourierdesignmatrix_dm(cls.psr.toas, freqs=cls.psr.freqs, nmodes=30)

        cls.B, _ = utils.create_fft_time_basis(cls.psr.toas, nnodes=30)

        cls.Bdm, _ = utils.create_fft_time_basis_dm(cls.psr.toas, freqs=cls.psr.freqs, nnodes=30)

        cls.Bchr, _ = utils.create_fft_time_basis_chromatic(cls.psr.toas, freqs=cls.psr.freqs, nnodes=30)

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

    def test_create_fft_time_basis(self, nk=30):
        """Check FFT interpolation design matrix shape."""

        msg = "FFT interpolation design matrix shape incorrect"
        assert self.B.shape == (4005, nk), msg

    def test_createfourierdesignmatrix_dm(self, nf=30):
        """Check DM-variation Fourier design matrix shape."""

        msg = "DM-variation Fourier design matrix shape incorrect"
        assert self.Fdm.shape == (4005, 2 * nf), msg

    def test_create_fft_time_basis_dm(self, nk=30):
        """Check FFT interpolation design matrix shape."""

        msg = "DM-variation FFT interpolation design matrix shape incorrect"
        assert self.Bdm.shape == (4005, nk), msg

    def test_create_fft_time_basis_chromatic(self, nk=30):
        """Check FFT interpolation design matrix shape."""

        msg = "DM-variation FFT interpolation design matrix shape incorrect"
        assert self.Bchr.shape == (4005, nk), msg

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

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions due to limited memory.")
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


@pytest.mark.skipif(not ASTROPY_INSTALLED, reason="Astropy required for native astrometry model")
class TestAstrometry(unittest.TestCase):
    """Astrometry design-matrix columns: finite differences + optional PINT/tempo2."""

    @classmethod
    def setUpClass(cls):
        cls.toas = np.linspace(53000.0, 58000.0, 48) * 86400.0
        cls.raj = 4.9533700839400492
        cls.decj = 0.16848694562363042
        cls.posepoch = float(np.mean(cls.toas))
        cls.dm, cls.names = utils.create_astrometry_timing_model(cls.toas, cls.raj, cls.decj, cls.posepoch)
        cls.mjd = (cls.toas * u.second).to(u.day)
        cls.rvec_m = utils.ssb_to_earth_vector(cls.mjd).to(u.m).value
        cls.c = ac.c.value
        cls.mas_to_rad = (1 * u.mas).to(u.rad).value

    def _n_hat(self, ra, dec, pmra_masyr=0.0, pmdec_masyr=0.0):
        dt_yr = (self.toas - self.posepoch) / (86400.0 * 365.25)
        # PMRA = mu_alpha* = d(alpha)/dt * cos(delta) [mas/yr]
        ra_t = ra + (pmra_masyr * dt_yr * self.mas_to_rad) / np.cos(dec)
        dec_t = dec + pmdec_masyr * dt_yr * self.mas_to_rad
        return np.column_stack([np.cos(dec_t) * np.cos(ra_t), np.cos(dec_t) * np.sin(ra_t), np.sin(dec_t)])

    def _roemer(self, ra, dec, px_mas=0.0, pmra_masyr=0.0, pmdec_masyr=0.0):
        n = self._n_hat(ra, dec, pmra_masyr=pmra_masyr, pmdec_masyr=pmdec_masyr)
        re_dot = np.sum(self.rvec_m * n, axis=1)
        tau = -re_dot / self.c
        if px_mas != 0.0:
            re_sqr = np.sum(self.rvec_m**2, axis=1)
            L_m = (1.0 / px_mas) * u.kpc.to(u.m)
            tau = tau + 0.5 * (re_sqr - re_dot**2) / (L_m * self.c)
        return tau

    def _assert_close_columns(self, analytic, finite, name, rtol=2e-4, atol=1e-10):
        # Ignore near-null samples when checking relative error.
        scale = np.maximum(np.abs(analytic), np.abs(finite))
        mask = scale > 1e-3 * np.max(scale)
        if not np.any(mask):
            mask = np.ones_like(analytic, dtype=bool)
        rel = np.abs(analytic[mask] - finite[mask]) / scale[mask]
        msg = f"{name}: max rel err {np.max(rel):.3e}, corr={np.corrcoef(analytic, finite)[0, 1]:.6f}"
        assert np.allclose(analytic[mask], finite[mask], rtol=rtol, atol=atol), msg
        assert np.corrcoef(analytic, finite)[0, 1] > 0.999, msg

    def test_parameter_names_and_shape(self):
        assert self.names == ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"]
        assert self.dm.shape == (len(self.toas), 5)
        # Units: PM columns must be tiny (s/(mas/yr)), not s^2/rad-scale.
        assert np.max(np.abs(self.dm[:, 2])) < 1e-3
        assert np.max(np.abs(self.dm[:, 3])) < 1e-3
        assert np.max(np.abs(self.dm[:, 0])) > 1.0

    def test_raj_decj_finite_difference(self):
        eps = 1e-9
        fd_ra = (self._roemer(self.raj + eps, self.decj) - self._roemer(self.raj - eps, self.decj)) / (2 * eps)
        fd_dec = (self._roemer(self.raj, self.decj + eps) - self._roemer(self.raj, self.decj - eps)) / (2 * eps)
        self._assert_close_columns(self.dm[:, 0], fd_ra, "RAJ")
        self._assert_close_columns(self.dm[:, 1], fd_dec, "DECJ")

    def test_pmra_pmdec_finite_difference(self):
        eps = 1e-3  # mas/yr
        fd_pmra = (
            self._roemer(self.raj, self.decj, pmra_masyr=eps) - self._roemer(self.raj, self.decj, pmra_masyr=-eps)
        ) / (2 * eps)
        fd_pmdec = (
            self._roemer(self.raj, self.decj, pmdec_masyr=eps) - self._roemer(self.raj, self.decj, pmdec_masyr=-eps)
        ) / (2 * eps)
        self._assert_close_columns(self.dm[:, 2], fd_pmra, "PMRA", rtol=5e-4, atol=1e-12)
        self._assert_close_columns(self.dm[:, 3], fd_pmdec, "PMDEC", rtol=5e-4, atol=1e-12)

    def test_px_finite_difference(self):
        eps = 1e-3  # mas
        fd_px = (self._roemer(self.raj, self.decj, px_mas=eps) - self._roemer(self.raj, self.decj, px_mas=0.0)) / eps
        self._assert_close_columns(self.dm[:, 4], fd_px, "PX", rtol=5e-4, atol=1e-12)

    def test_spindown_polynomial_basis(self):
        M, names = utils.create_spindown_timing_model(self.toas, order=2, pepoch=self.posepoch)
        assert names == ["Offset", "Poly1", "Poly2"]
        assert M.shape == (len(self.toas), 3)
        assert np.allclose(M[:, 0], 1.0)
        t_yr = (self.toas - self.posepoch) / (86400.0 * 365.25)
        assert np.allclose(M[:, 1], t_yr)
        assert np.allclose(M[:, 2], t_yr**2)

    def test_matches_pint_formula_units(self):
        """Reproduce PINT's published analytic expressions on the same Earth vectors.

        This does not import PINT (so it always runs). The optional
        ``test_against_pint_designmatrix`` check exercises a live PINT model
        when the package is installed.
        """
        from astropy.time import Time

        ssb_obs = self.rvec_m * u.m
        ssb_psr = np.array(
            [
                np.cos(self.decj) * np.cos(self.raj),
                np.cos(self.decj) * np.sin(self.raj),
                np.sin(self.decj),
            ]
        )
        ssb_obs_r = np.sqrt(np.sum(ssb_obs**2, axis=1))
        earth_dec = np.arctan2(ssb_obs[:, 2], np.sqrt(ssb_obs[:, 0] ** 2 + ssb_obs[:, 1] ** 2))
        earth_ra = np.arctan2(ssb_obs[:, 1], ssb_obs[:, 0])
        pe = Time(self.posepoch / 86400.0, format="mjd", scale="tdb")
        te = (self.toas / 86400.0) * u.day - pe.tdb.mjd_long * u.day

        geom_ra = np.cos(earth_dec) * np.cos(self.decj * u.rad) * np.sin(self.raj * u.rad - earth_ra)
        pint_ra = (ssb_obs_r * geom_ra / (ac.c * u.radian)).decompose(u.si.bases).to_value(u.s / u.rad)

        geom_dec = np.cos(earth_dec) * np.sin(self.decj * u.rad) * np.cos(self.raj * u.rad - earth_ra) - np.sin(
            earth_dec
        ) * np.cos(self.decj * u.rad)
        pint_dec = (ssb_obs_r * geom_dec / (ac.c * u.radian)).decompose(u.si.bases).to_value(u.s / u.rad)

        geom_pmra = np.cos(earth_dec) * np.sin(self.raj * u.rad - earth_ra)
        pint_pmra = (
            (ssb_obs_r * geom_pmra * te / (ac.c * u.radian) * u.mas / u.year).decompose(u.si.bases) / (u.mas / u.year)
        ).to_value(u.s / (u.mas / u.year))

        geom_pmdec = np.cos(earth_dec) * np.sin(self.decj * u.rad) * np.cos(self.raj * u.rad - earth_ra) - np.cos(
            self.decj * u.rad
        ) * np.sin(earth_dec)
        pint_pmdec = (
            (ssb_obs_r * geom_pmdec * te / (ac.c * u.radian) * u.mas / u.year).decompose(u.si.bases) / (u.mas / u.year)
        ).to_value(u.s / (u.mas / u.year))

        in_psr_obs = np.sum(ssb_obs * ssb_psr, axis=1)
        px_r = np.sqrt(ssb_obs_r**2 - in_psr_obs**2)
        # PINT: multiply by (mas/radian), decompose, then divide by mas → s/mas
        pint_px = np.asarray(
            (0.5 * (px_r**2 / (u.AU * ac.c)) * (u.mas / u.radian)).decompose(u.si.bases) / u.mas,
            dtype=float,
        )

        self._assert_close_columns(self.dm[:, 0], pint_ra, "RAJ-PINT-formula", rtol=1e-10, atol=1e-12)
        self._assert_close_columns(self.dm[:, 1], pint_dec, "DECJ-PINT-formula", rtol=1e-10, atol=1e-12)
        self._assert_close_columns(self.dm[:, 2], pint_pmra, "PMRA-PINT-formula", rtol=1e-10, atol=1e-18)
        self._assert_close_columns(self.dm[:, 3], pint_pmdec, "PMDEC-PINT-formula", rtol=1e-10, atol=1e-18)
        self._assert_close_columns(self.dm[:, 4], pint_px, "PX-PINT-formula", rtol=1e-10, atol=1e-18)

    @pytest.mark.skipif(not PINT_INSTALLED, reason="PINT not installed")
    def test_against_pint_designmatrix(self):
        """Live PINT design matrix: signed correlation for equatorial astrometry cols."""
        from pint.models import get_model_and_toas

        # 1713 test set is equatorial (RAJ/DECJ); B1855 9yr is ecliptic.
        model, toas = get_model_and_toas(
            datadir + "/1713.Sep.T2.par",
            datadir + "/1713.Sep.T2.tim",
            allow_tcb=True,
            allow_T2=True,
        )
        if "AstrometryEquatorial" not in model.components:
            self.skipTest("Test parfile is not equatorial")

        M, params, _units = model.designmatrix(toas)
        # PINT designmatrix rows follow TOA order; barycentric times in seconds.
        bat = np.array(model.get_barycentric_toas(toas).value, dtype=float) * 86400.0
        raj = float(model.RAJ.quantity.to_value(u.rad))
        decj = float(model.DECJ.quantity.to_value(u.rad))
        posepoch = float(model.POSEPOCH.quantity.tdb.mjd_long) * 86400.0
        dm, names = utils.create_astrometry_timing_model(bat, raj, decj, posepoch)

        for pname in names:
            if pname not in params:
                continue
            ours = dm[:, names.index(pname)]
            pint_col = np.asarray(M[:, list(params).index(pname)], dtype=float)
            # PINT residual design matrix may absorb F0 scaling on some builds;
            # require strong signed correlation of the annual geometry.
            corr = np.corrcoef(ours, pint_col)[0, 1]
            assert corr > 0.95, f"{pname}: correlation with PINT designmatrix is {corr}"

    @pytest.mark.skipif(not LIBSTEMPO_INSTALLED, reason="libstempo not installed")
    def test_against_tempo2_signed_shape(self):
        """Signed comparison of annual geometry vs tempo2 (Earth vs observatory).

        Run in a **subprocess**: libstempo is known to segfault if PINT has
        already constructed a timing model in the same process (see enterprise
        CI notes / long-lived Pulsar() constructions).
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            f"""
            import numpy as np
            from enterprise.pulsar import Pulsar
            from enterprise.signals import utils

            datadir = {datadir!r}
            psr = Pulsar(
                datadir + "/1713.Sep.T2.par",
                datadir + "/1713.Sep.T2.tim",
                timing_package="tempo2",
                drop_t2pulsar=False,
            )
            Mmat = psr.t2pulsar.designmatrix(fixunits=False, fixsigns=True, incoffset=True)
            posepoch = psr.t2pulsar["POSEPOCH"].val * 86400.0
            dm, names = utils.create_astrometry_timing_model(
                psr.toas, psr._raj, psr._decj, posepoch
            )
            t2names = ("Offset",) + psr.t2pulsar.pars(which="fit")
            for pname in names:
                if pname not in t2names:
                    continue
                ours = dm[:, names.index(pname)]
                t2 = Mmat[psr._isort, t2names.index(pname)]
                corr = np.corrcoef(ours, t2)[0, 1]
                assert corr > 0.95, f"{{pname}}: correlation with tempo2 is {{corr}}"
                if pname in ("PMRA", "PMDEC"):
                    continue
                rms_ratio = np.std(ours) / (np.std(t2) + 1e-30)
                assert 0.5 < rms_ratio < 2.0, f"{{pname}}: RMS ratio vs tempo2 is {{rms_ratio}}"
            print("OK")
            """
        )
        env = dict(**os.environ)
        # Prefer the enterprise tree under test.
        ent_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = ent_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert proc.returncode == 0, f"tempo2 subprocess failed:\n{proc.stdout}\n{proc.stderr}"
