#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_white_signals
----------------------------------

Tests for white signal modules.
"""


import numpy as np
import pytest
import scipy.linalg as sl

try:
    import libstempo as t2
except (ImportError, RuntimeError):
    t2 = None

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals, parameter, selections, utils, white_signals
from enterprise.signals.selections import Selection
from tests.enterprise_test_data import datadir


class Woodbury(object):
    def __init__(self, N, U, J):
        self.N = N
        self.U = U
        self.J = J

    def solve(self, other):
        if other.ndim == 1:
            Nx = np.array(other / self.N)
        elif other.ndim == 2:
            Nx = np.array(other / self.N[:, None])
        UNx = np.dot(self.U.T, Nx)

        Sigma = np.diag(1 / self.J) + np.dot(self.U.T, self.U / self.N[:, None])
        cf = sl.cho_factor(Sigma)
        if UNx.ndim == 1:
            tmp = np.dot(self.U, sl.cho_solve(cf, UNx)) / self.N
        else:
            tmp = np.dot(self.U, sl.cho_solve(cf, UNx)) / self.N[:, None]
        return Nx - tmp

    def logdet(self):
        Sigma = np.diag(1 / self.J) + np.dot(self.U.T, self.U / self.N[:, None])
        cf = sl.cho_factor(Sigma)
        ld = np.sum(np.log(self.N)) + np.sum(np.log(self.J))
        ld += np.sum(2 * np.log(np.diag(cf[0])))
        return ld


@pytest.fixture(scope="module", params=["tempo2", "pint"])
def psr(request):
    timing_package = request.param
    if t2 is None and timing_package == "tempo2":
        pytest.skip("TEMPO2 not available")
    if timing_package == "pint":
        # Why specify the ephemeris for PINT? Faster?
        return Pulsar(
            f"{datadir}/B1855+09_NANOGrav_9yv1.gls.par",
            f"{datadir}/B1855+09_NANOGrav_9yv1.tim",
            ephem="DE430",
            timing_package=timing_package,
        )
    else:
        return Pulsar(
            f"{datadir}/B1855+09_NANOGrav_9yv1.gls.par",
            f"{datadir}/B1855+09_NANOGrav_9yv1.tim",
            timing_package=timing_package,
        )


@pytest.fixture(
    scope="module",
    params=["tempo2", pytest.param("pint", marks=pytest.mark.xfail(reason="PINT does not support T2 model"))],
)
def ipsr(request):
    timing_package = request.param
    if t2 is None and timing_package == "tempo2":
        pytest.skip("TEMPO2 not available")
    # FIXME: this was supposed to be tested with PINT but PINT does not support
    # the T2 timing model; a typo made this problem invisible.
    # IPTA-like pulsar
    if timing_package == "pint":
        # Why specify the ephemeris for PINT? Faster?
        return Pulsar(
            f"{datadir}/1713.Sep.T2.par",
            f"{datadir}/1713.Sep.T2.tim",
            ephem="DE430",
            timing_package=timing_package,
        )
    else:
        return Pulsar(
            f"{datadir}/1713.Sep.T2.par",
            f"{datadir}/1713.Sep.T2.tim",
            timing_package=timing_package,
        )


def test_efac(psr):
    """Test that efac signal returns correct covariance."""
    # set up signal and parameters
    efac = parameter.Uniform(0.1, 5)
    ef = white_signals.MeasurementNoise(efac=efac)
    efm = ef(psr)

    # parameters
    efac = 1.5
    params = {"B1855+09_efac": efac}

    # correct value
    nvec0 = efac**2 * psr.toaerrs**2

    # test
    msg = "EFAC covariance incorrect."
    assert np.all(efm.get_ndiag(params) == nvec0), msg


def test_efac_backend(psr):
    """Test that backend-efac signal returns correct covariance."""
    # set up signal and parameters
    efac = parameter.Uniform(0.1, 5)
    selection = Selection(selections.by_backend)
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    efm = ef(psr)

    # parameters
    efacs = [1.3, 1.4, 1.5, 1.6]
    params = {
        "B1855+09_430_ASP_efac": efacs[0],
        "B1855+09_430_PUPPI_efac": efacs[1],
        "B1855+09_L-wide_ASP_efac": efacs[2],
        "B1855+09_L-wide_PUPPI_efac": efacs[3],
    }

    # correct value
    flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
    nvec0 = np.zeros_like(psr.toas)
    for ct, flag in enumerate(np.unique(flags)):
        ind = flag == psr.backend_flags
        nvec0[ind] = efacs[ct] ** 2 * psr.toaerrs[ind] ** 2

    # test
    msg = "EFAC covariance incorrect."
    assert np.all(efm.get_ndiag(params) == nvec0), msg


def test_equad(psr):
    """Test that the deprecated EquadNoise is not available."""

    with pytest.raises(NotImplementedError):
        white_signals.EquadNoise()


def test_tnequad(psr):
    """Test that tnequad signal returns correct covariance."""
    # set up signal and parameters
    tnequad = parameter.Uniform(-10, -5)
    eq = white_signals.TNEquadNoise(log10_tnequad=tnequad)
    eqm = eq(psr)

    # parameters
    tnequad = -6.4
    params = {"B1855+09_log10_tnequad": tnequad}

    # correct value
    nvec0 = 10 ** (2 * tnequad) * np.ones_like(psr.toas)

    # test
    msg = "TNEQUAD covariance incorrect."
    assert np.all(eqm.get_ndiag(params) == nvec0), msg


def test_tnequad_backend(psr):
    """Test that backend-equad signal returns correct covariance."""
    # set up signal and parameters
    tnequad = parameter.Uniform(-10, -5)
    selection = Selection(selections.by_backend)
    eq = white_signals.TNEquadNoise(log10_tnequad=tnequad, selection=selection)
    eqm = eq(psr)

    # parameters
    tnequads = [-6.1, -6.2, -6.3, -6.4]
    params = {
        "B1855+09_430_ASP_log10_tnequad": tnequads[0],
        "B1855+09_430_PUPPI_log10_tnequad": tnequads[1],
        "B1855+09_L-wide_ASP_log10_tnequad": tnequads[2],
        "B1855+09_L-wide_PUPPI_log10_tnequad": tnequads[3],
    }

    # correct value
    flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
    nvec0 = np.zeros_like(psr.toas)
    for ct, flag in enumerate(np.unique(flags)):
        ind = flag == psr.backend_flags
        nvec0[ind] = 10 ** (2 * tnequads[ct]) * np.ones(np.sum(ind))

    # test
    msg = "TNEQUAD covariance incorrect."
    assert np.all(eqm.get_ndiag(params) == nvec0), msg


def test_add_efac_tnequad(psr):
    """Test that addition of efac and tnequad signal returns
    correct covariance.
    """
    # set up signals
    efac = parameter.Uniform(0.1, 5)
    ef = white_signals.MeasurementNoise(efac=efac)
    tnequad = parameter.Uniform(-10, -5)
    eq = white_signals.TNEquadNoise(log10_tnequad=tnequad)
    s = ef + eq
    m = s(psr)

    # set parameters
    efac = 1.5
    tnequad = -6.4
    params = {"B1855+09_efac": efac, "B1855+09_log10_tnequad": tnequad}

    # correct value
    nvec0 = efac**2 * psr.toaerrs**2
    nvec0 += 10 ** (2 * tnequad) * np.ones_like(psr.toas)

    # test
    msg = "EFAC/TNEQUAD covariance incorrect."
    assert np.all(m.get_ndiag(params) == nvec0), msg


def test_add_efac_tnequad_backend(psr):
    """Test that addition of efac-backend and tnequad-backend signal returns
    correct covariance.
    """
    selection = Selection(selections.by_backend)

    efac = parameter.Uniform(0.1, 5)
    tnequad = parameter.Uniform(-10, -5)
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.TNEquadNoise(log10_tnequad=tnequad, selection=selection)
    s = ef + eq
    m = s(psr)

    # set parameters
    efacs = [1.3, 1.4, 1.5, 1.6]
    tnequads = [-6.1, -6.2, -6.3, -6.4]
    params = {
        "B1855+09_430_ASP_efac": efacs[0],
        "B1855+09_430_PUPPI_efac": efacs[1],
        "B1855+09_L-wide_ASP_efac": efacs[2],
        "B1855+09_L-wide_PUPPI_efac": efacs[3],
        "B1855+09_430_ASP_log10_tnequad": tnequads[0],
        "B1855+09_430_PUPPI_log10_tnequad": tnequads[1],
        "B1855+09_L-wide_ASP_log10_tnequad": tnequads[2],
        "B1855+09_L-wide_PUPPI_log10_tnequad": tnequads[3],
    }

    # correct value
    flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
    nvec0 = np.zeros_like(psr.toas)
    for ct, flag in enumerate(np.unique(flags)):
        ind = flag == psr.backend_flags
        nvec0[ind] = efacs[ct] ** 2 * psr.toaerrs[ind] ** 2
        nvec0[ind] += 10 ** (2 * tnequads[ct]) * np.ones(np.sum(ind))

    logdet = np.sum(np.log(nvec0))

    # test
    msg = "EFAC/TNEQUAD covariance incorrect."
    assert np.all(m.get_ndiag(params) == nvec0), msg

    msg = "EFAC/TNEQUAD logdet incorrect."
    N = m.get_ndiag(params)
    assert np.allclose(N.solve(psr.residuals, logdet=True)[1], logdet, rtol=1e-10), msg

    msg = "EFAC/TNEQUAD D1 solve incorrect."
    assert np.allclose(N.solve(psr.residuals), psr.residuals / nvec0, rtol=1e-10), msg

    msg = "EFAC/TNEQUAD 1D1 solve incorrect."
    assert np.allclose(
        N.solve(psr.residuals, left_array=psr.residuals),
        np.dot(psr.residuals / nvec0, psr.residuals),
        rtol=1e-10,
    ), msg

    msg = "EFAC/TNEQUAD 2D1 solve incorrect."
    T = psr.Mmat
    assert np.allclose(N.solve(psr.residuals, left_array=T), np.dot(T.T, psr.residuals / nvec0), rtol=1e-10), msg

    msg = "EFAC/TNEQUAD 2D2 solve incorrect."
    assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, T / nvec0[:, None]), rtol=1e-10), msg


def test_efac_equad_combined_backend(psr):
    """Test that the combined EFAC + EQUAD noise (tempo/tempo2/pint definition)
    returns the correct covariance.
    """
    selection = Selection(selections.by_backend)

    efac = parameter.Uniform(0.1, 5)
    t2equad = parameter.Uniform(-10, -5)
    efq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
    m = efq(psr)

    # set parameters
    efacs = [1.3, 1.4, 1.5, 1.6]
    equads = [-6.1, -6.2, -6.3, -6.4]
    params = {
        "B1855+09_430_ASP_efac": efacs[0],
        "B1855+09_430_PUPPI_efac": efacs[1],
        "B1855+09_L-wide_ASP_efac": efacs[2],
        "B1855+09_L-wide_PUPPI_efac": efacs[3],
        "B1855+09_430_ASP_log10_t2equad": equads[0],
        "B1855+09_430_PUPPI_log10_t2equad": equads[1],
        "B1855+09_L-wide_ASP_log10_t2equad": equads[2],
        "B1855+09_L-wide_PUPPI_log10_t2equad": equads[3],
    }

    # correct value
    flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
    nvec0 = np.zeros_like(psr.toas)
    for ct, flag in enumerate(np.unique(flags)):
        ind = flag == psr.backend_flags
        nvec0[ind] = efacs[ct] ** 2 * (psr.toaerrs[ind] ** 2 + 10 ** (2 * equads[ct]) * np.ones(np.sum(ind)))

    logdet = np.sum(np.log(nvec0))

    # test
    msg = "EFAC+EQUAD covariance incorrect."
    assert np.all(m.get_ndiag(params) == nvec0), msg

    msg = "EFAC+EQUAD logdet incorrect."
    N = m.get_ndiag(params)
    assert np.allclose(N.solve(psr.residuals, logdet=True)[1], logdet, rtol=1e-10), msg

    msg = "EFAC+EQUAD D1 solve incorrect."
    assert np.allclose(N.solve(psr.residuals), psr.residuals / nvec0, rtol=1e-10), msg

    msg = "EFAC+EQUAD 1D1 solve incorrect."
    assert np.allclose(
        N.solve(psr.residuals, left_array=psr.residuals),
        np.dot(psr.residuals / nvec0, psr.residuals),
        rtol=1e-10,
    ), msg

    msg = "EFAC+EQUAD 2D1 solve incorrect."
    T = psr.Mmat
    assert np.allclose(N.solve(psr.residuals, left_array=T), np.dot(T.T, psr.residuals / nvec0), rtol=1e-10), msg

    msg = "EFAC+EQUAD 2D2 solve incorrect."
    assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, T / nvec0[:, None]), rtol=1e-10), msg


@pytest.mark.parametrize("method", ["sparse", "sherman-morrison", "block"])
def test_ecorr(psr, method):
    """Test of sparse/sherman-morrison ecorr signal and solve methods."""
    selection = Selection(selections.by_backend)

    efac = parameter.Uniform(0.1, 5)
    ecorr = parameter.Uniform(-10, -5)
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection, method=method)
    tm = gp_signals.TimingModel()
    s = ef + ec + tm
    m = s(psr)

    # set parameters
    efacs = [1.3, 1.4, 1.5, 1.6]
    ecorrs = [-6.1, -6.2, -6.3, -6.4]
    params = {
        "B1855+09_430_ASP_efac": efacs[0],
        "B1855+09_430_PUPPI_efac": efacs[1],
        "B1855+09_L-wide_ASP_efac": efacs[2],
        "B1855+09_L-wide_PUPPI_efac": efacs[3],
        "B1855+09_430_ASP_log10_ecorr": ecorrs[0],
        "B1855+09_430_PUPPI_log10_ecorr": ecorrs[1],
        "B1855+09_L-wide_ASP_log10_ecorr": ecorrs[2],
        "B1855+09_L-wide_PUPPI_log10_ecorr": ecorrs[3],
    }

    # get EFAC Nvec
    flags = ["430_ASP", "430_PUPPI", "L-wide_ASP", "L-wide_PUPPI"]
    nvec0 = np.zeros_like(psr.toas)
    for ct, flag in enumerate(np.unique(flags)):
        ind = flag == psr.backend_flags
        nvec0[ind] = efacs[ct] ** 2 * psr.toaerrs[ind] ** 2

    # get the basis
    bflags = psr.backend_flags
    Umats = []
    for flag in np.unique(bflags):
        mask = bflags == flag
        Umats.append(utils.create_quantization_matrix(psr.toas[mask], nmin=2)[0])
    nepoch = sum(U.shape[1] for U in Umats)
    U = np.zeros((len(psr.toas), nepoch))
    jvec = np.zeros(nepoch)
    netot = 0
    for ct, flag in enumerate(np.unique(bflags)):
        mask = bflags == flag
        nn = Umats[ct].shape[1]
        U[mask, netot : nn + netot] = Umats[ct]
        jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ct])
        netot += nn

    # get covariance matrix
    wd = Woodbury(nvec0, U, jvec)

    # test
    msg = f"EFAC/ECORR {method} logdet incorrect."
    N = m.get_ndiag(params)
    assert np.allclose(N.solve(psr.residuals, logdet=True)[1], wd.logdet(), rtol=1e-10), msg

    msg = f"EFAC/ECORR {method} D1 solve incorrect."
    assert np.allclose(N.solve(psr.residuals), wd.solve(psr.residuals), rtol=1e-10), msg

    msg = f"EFAC/ECORR {method} 1D1 solve incorrect."
    assert np.allclose(
        N.solve(psr.residuals, left_array=psr.residuals),
        np.dot(psr.residuals, wd.solve(psr.residuals)),
        rtol=1e-10,
    ), msg

    msg = f"EFAC/ECORR {method} 2D1 solve incorrect."
    T = m.get_basis()
    assert np.allclose(N.solve(psr.residuals, left_array=T), np.dot(T.T, wd.solve(psr.residuals)), rtol=1e-10), msg

    msg = f"EFAC/ECORR {method} 2D2 solve incorrect."
    assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, wd.solve(T)), rtol=1e-10), msg


@pytest.mark.parametrize("method", ["sparse", "sherman-morrison", "block"])
def test_ecorr_ipta(ipsr, method):
    """Test of sparse/sherman-morrison ecorr signal and solve methods."""
    selection = Selection(selections.nanograv_backends)

    efac = parameter.Uniform(0.1, 5)
    ecorr = parameter.Uniform(-10, -5)
    ef = white_signals.MeasurementNoise(efac=efac)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection, method=method)
    tm = gp_signals.TimingModel()
    s = ef + ec + tm
    m = s(ipsr)

    # set parameters
    efacs = [1.3]
    ecorrs = [-6.1, -6.2, -6.3, -6.4, -7.2, -8.4, -7.1, -7.9]
    params = {
        "J1713+0747_efac": efacs[0],
        "J1713+0747_ASP-L_log10_ecorr": ecorrs[0],
        "J1713+0747_ASP-S_log10_ecorr": ecorrs[1],
        "J1713+0747_GASP-8_log10_ecorr": ecorrs[2],
        "J1713+0747_GASP-L_log10_ecorr": ecorrs[3],
        "J1713+0747_GUPPI-8_log10_ecorr": ecorrs[4],
        "J1713+0747_GUPPI-L_log10_ecorr": ecorrs[5],
        "J1713+0747_PUPPI-L_log10_ecorr": ecorrs[6],
        "J1713+0747_PUPPI-S_log10_ecorr": ecorrs[7],
    }

    # get EFAC Nvec
    nvec0 = efacs[0] ** 2 * ipsr.toaerrs**2

    # get the basis
    flags = ["ASP-L", "ASP-S", "GASP-8", "GASP-L", "GUPPI-8", "GUPPI-L", "PUPPI-L", "PUPPI-S"]
    bflags = ipsr.backend_flags
    Umats = []
    for flag in np.unique(bflags):
        if flag in flags:
            mask = bflags == flag
            Umats.append(utils.create_quantization_matrix(ipsr.toas[mask], nmin=2)[0])
    nepoch = sum(U.shape[1] for U in Umats)
    U = np.zeros((len(ipsr.toas), nepoch))
    jvec = np.zeros(nepoch)
    netot, ct = 0, 0
    for flag in np.unique(bflags):
        if flag in flags:
            mask = bflags == flag
            nn = Umats[ct].shape[1]
            U[mask, netot : nn + netot] = Umats[ct]
            jvec[netot : nn + netot] = 10 ** (2 * ecorrs[ct])
            netot += nn
            ct += 1

    # get covariance matrix
    wd = Woodbury(nvec0, U, jvec)

    # test
    msg = f"EFAC/ECORR {method} logdet incorrect."
    N = m.get_ndiag(params)
    assert np.allclose(N.solve(ipsr.residuals, logdet=True)[1], wd.logdet(), rtol=1e-8), msg

    msg = f"EFAC/ECORR {method} D1 solve incorrect."
    assert np.allclose(N.solve(ipsr.residuals), wd.solve(ipsr.residuals), rtol=1e-8), msg

    msg = f"EFAC/ECORR {method} 1D1 solve incorrect."
    assert np.allclose(
        N.solve(ipsr.residuals, left_array=ipsr.residuals),
        np.dot(ipsr.residuals, wd.solve(ipsr.residuals)),
        rtol=1e-8,
    ), msg

    msg = f"EFAC/ECORR {method} 2D1 solve incorrect."
    T = m.get_basis()
    assert np.allclose(N.solve(ipsr.residuals, left_array=T), np.dot(T.T, wd.solve(ipsr.residuals)), rtol=1e-8), msg

    msg = f"EFAC/ECORR {method} 2D2 solve incorrect."
    assert np.allclose(N.solve(T, left_array=T), np.dot(T.T, wd.solve(T)), rtol=1e-8), msg
