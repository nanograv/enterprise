# gp_bases.py
"""Utilities module containing various useful
functions for use in other modules.
"""

import numpy as np
from enterprise.signals.parameter import function

######################################
# Fourier-basis signal functions #####
######################################

__all__ = [
    "createfourierdesignmatrix_red",
    "create_fft_time_basis",
    "createfourierdesignmatrix_dm",
    "createfourierdesignmatrix_dm_tn",
    "createfourierdesignmatrix_env",
    "createfourierdesignmatrix_ephem",
    "createfourierdesignmatrix_eph",
    "createfourierdesignmatrix_chromatic",
    "createfourierdesignmatrix_general",
]


@function
def createfourierdesignmatrix_red(
    toas, nmodes=30, Tspan=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None, pseed=None
):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param pseed: option to provide phase shift seed
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # if requested, add random phase shift to basis functions
    if pshift or pseed is not None:
        if pseed is not None:
            # use the first toa to make a different seed for every pulsar
            seed = int(toas[0] / 17) + int(pseed)
            np.random.seed(seed)

        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    else:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:, ::2] = np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    return F, Ffreqs


@function
def create_fft_time_basis(toas, nmodes=30, Tspan=None, start_time=None):
    """
    Construct coarse time-domain design matrix from eq 11 of Chrisostomi et al., 2025
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param start_time: option to set some other start epoch of basis

    :return B: coarse time-domain design matrix
    :return t_coarse: timestamps of coarse time grid
    """
    if start_time is None:
        start_time = np.min(toas)
    else:
        if start_time > np.min(toas):
            raise ValueError("Coarse time basis start must be earlier than earliest TOA.")

    if Tspan is None:
        Tspan = np.max(toas) - start_time
    else:
        if start_time + Tspan < np.max(toas):
            raise ValueError("Coarse time basis end must be later than latest TOA.")

    t_fine = toas
    t_coarse = np.linspace(start_time, start_time + Tspan, nmodes)
    dt_coarse = t_coarse[1] - t_coarse[0]

    idx = np.arange(len(t_fine))
    idy = np.searchsorted(t_coarse, t_fine)
    idy = np.clip(idy, 1, nmodes - 1)

    Bmat = np.zeros((len(t_fine), len(t_coarse)), "d")

    Bmat[idx, idy] = (t_fine - t_coarse[idy - 1]) / dt_coarse
    Bmat[idx, idy - 1] = (t_coarse[idy] - t_fine) / dt_coarse

    return Bmat, t_coarse


@function
def createfourierdesignmatrix_dm(
    toas, freqs, nmodes=30, Tspan=None, pshift=False, fref=1400, logf=False, fmin=None, fmax=None, modes=None
):
    """
    Construct DM-variation fourier design matrix. Current
    normalization expresses DM signal as a deviation [seconds]
    at fref [MHz]

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param pshift: option to add random phase shift
    :param fref: reference frequency [MHz]
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, pshift=pshift, modes=modes
    )

    # compute the DM-variation vectors
    Dm = (fref / freqs) ** 2

    return F * Dm[:, None], Ffreqs


@function
def createfourierdesignmatrix_dm_tn(
    toas, freqs, nmodes=30, Tspan=None, pshift=False, fref=1400, logf=False, fmin=None, fmax=None, idx=2, modes=None
):
    """
    Construct DM-variation fourier design matrix. Current
    normalization expresses DM signal as a deviation [seconds]
    at fref [MHz]

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param pshift: option to add random phase shift
    :param fref: reference frequency [MHz]
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param idx: index of the radio frequency dependence
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, pshift=pshift, modes=modes
    )

    # compute the DM-variation vectors in the temponest normalization
    # amplitude normalization: sqrt(12)*pi, scaling to 1 MHz from 1400 MHz, DM constant: 2.41e-4
    Dm = (fref / freqs) ** idx * np.sqrt(12) * np.pi / 1400 / 1400 / 2.41e-4

    return F * Dm[:, None], Ffreqs


@function
def createfourierdesignmatrix_env(
    toas,
    log10_Amp=-7,
    log10_Q=np.log10(300),
    t0=53000 * 86400,
    nmodes=30,
    Tspan=None,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
):
    """
    Construct fourier design matrix with gaussian envelope.

    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param log10_Amp: log10 of the Amplitude [s]
    :param t0: mean of gaussian envelope [s]
    :param log10_Q: log10 of standard deviation of gaussian envelope [days]
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix with gaussian envelope
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes
    )

    # compute gaussian envelope
    A = 10**log10_Amp
    Q = 10**log10_Q * 86400
    env = A * np.exp(-((toas - t0) ** 2) / 2 / Q**2)
    return F * env[:, None], Ffreqs


@function
def createfourierdesignmatrix_ephem(toas, pos, nmodes=30, Tspan=None):
    """
    Construct ephemeris perturbation Fourier design matrix and frequencies.
    The matrix contains nmodes*6 columns, ordered as by frequency first,
    Cartesian coordinate second:

    sin(f0) [x], sin(f0) [y], sin(f0) [z],
    cos(f0) [x], cos(f0) [y], cos(f0) [z],
    sin(f1) [x], sin(f1) [y], sin(f1) [z], ...

    The corresponding frequency vector repeats every entry six times.
    This design matrix should be used with monopole_orf and with
    a powerlaw that specifies components=6.

    :param toas: vector of time series in seconds
    :param pos: pulsar position as Cartesian vector
    :param nmodes: number of Fourier coefficients
    :param Tspan: Tspan used to define Fourier bins

    :return: F: Fourier design matrix of shape (len(toas),6*nmodes)
    :return: f: Sampling frequencies (6*nmodes)
    """

    F0, F0f = createfourierdesignmatrix_red(toas, nmodes=nmodes, Tspan=Tspan)

    F1 = np.zeros((len(toas), nmodes, 2, 3), "d")
    F1[:, :, 0, :] = F0[:, 0::2, np.newaxis]
    F1[:, :, 1, :] = F0[:, 1::2, np.newaxis]

    # verify this is the scalar product we want
    F1 *= pos

    F1f = np.zeros((nmodes, 2, 3), "d")
    F1f[:, :, :] = F0f[::2, np.newaxis, np.newaxis]

    return F1.reshape((len(toas), nmodes * 6)), F1f.reshape((nmodes * 6,))


def createfourierdesignmatrix_eph(
    t, nmodes, phi, theta, freq=False, Tspan=None, logf=False, fmin=None, fmax=None, modes=None
):
    raise NotImplementedError(
        "createfourierdesignmatrix_eph was removed, " + "and replaced with createfourierdesignmatrix_ephem"
    )


@function
def createfourierdesignmatrix_chromatic(
    toas, freqs, nmodes=30, Tspan=None, logf=False, fmin=None, fmax=None, idx=4, modes=None
):
    """
    Construct Scattering-variation fourier design matrix.

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param idx: Index of chromatic effects
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: Chromatic-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes
    )

    # compute the DM-variation vectors
    Dm = (1400 / freqs) ** idx

    return F * Dm[:, None], Ffreqs


@function
def createfourierdesignmatrix_general(
    toas,
    freqs,
    flags,
    flagname="group",
    flagval=None,
    idx=None,
    tndm=False,
    nmodes=30,
    Tspan=None,
    psrTspan=True,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
    pshift=None,
    pseed=None,
):
    """
    Construct fourier design matrix with possibility of adding selection and/or chromatic index envelope.

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param flags: Flags from timfiles
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param psrTspan: option to use pulsar time span. Used only if sub-group of ToAs is chosen
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param log10_Amp: log10 of the Amplitude [s]
    :param idx: Index of chromatic effects
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """
    if flagval and not psrTspan:
        sel_toas = toas[np.where(flags[flagname] == flagval)]
        Tspan = sel_toas.max() - sel_toas.min()

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes, pshift=pshift, pseed=pseed
    )

    # compute the chromatic-variation vectors
    if idx:
        if tndm:
            chrom_fac = (1400 / freqs) ** idx * np.sqrt(12) * np.pi / 1400 / 1400 / 2.41e-4
        else:
            chrom_fac = (1400 / freqs) ** idx
        F *= chrom_fac[:, None]

    # compute the mask for the selection
    if flagval:
        F *= np.array([flags[flagname] == flagval] * F.shape[1]).T

    return F, Ffreqs
