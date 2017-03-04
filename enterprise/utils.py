#utils.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import enterprise.constants as const


def create_stabletimingdesignmatrix(designmat, fastDesign=True):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param designmat: Pulsar timing model design matrix
    :param fastDesign: Stabilize the design matrix the fast way [True]

    :return: Mm: Stabilized timing model design matrix
    """

    Mm = designmat.copy()

    if fastDesign:

        norm = np.sqrt(np.sum(Mm ** 2, axis=0))
        Mm /= norm

    else:

        u, s, v = np.linalg.svd(Mm)
        Mm = u[:,:len(s)]

    return Mm


def createfourierdesignmatrix_red(t, nmodes, freq=False, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  pshift=False):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift

    :return: F: fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    :return: ranphase: Phase offsets applied to basis functions
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    # add random phase shift to basis functions
    if pshift:
        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    elif not pshift:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # The sine/cosine modes
    if pshift:
        F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:] +
                          ranphase[None,:])
        F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:] +
                           ranphase[None,:])
    elif not pshift:
        F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:])
        F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:])

    if freq:
        return F, Ffreqs, ranphase
    else:
        return F, ranphase


def createfourierdesignmatrix_dm(t, nmodes, ssbfreqs, freq=False,
                                 Tspan=None, logf=False, fmin=None,
                                 fmax=None):

    """
    Construct DM-variation fourier design matrix.

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param ssbfreqs: radio frequencies of observations [MHz]
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # compute the DM-variation vectors
    Dm = 1.0/(const.DM_K * ssbfreqs**2.0 * 1e12)

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:]) * Dm[:,None]
    F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:]) * Dm[:,None]

    if freq:
        return F, Ffreqs
    else:
        return F


def createfourierdesignmatrix_eph(t, nmodes, phi, theta, freq=False,
                                  Tspan=None, logf=False, fmin=None,
                                  fmax=None):

    """
    Construct ephemeris fourier design matrix.

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param phi: azimuthal coordinate of pulsar
    :param theta: polar coordinate of pulsar
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: Fx: x-axis ephemeris fourier design matrix
    :return: Fy: y-axis ephemeris fourier design matrix
    :return: Fz: z-axis ephemeris fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    """

    N = len(t)
    Fx = np.zeros((N, 2*nmodes))
    Fy = np.zeros((N, 2*nmodes))
    Fz = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # define the pulsar position vector
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    # The sine/cosine modes
    Fx[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:])
    Fx[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:])

    Fy = Fx.copy()
    Fz = Fx.copy()

    Fx *= x
    Fy *= y
    Fz *= z

    if freq:
        return Fx, Fy, Fz, Ffreqs
    else:
        return Fx, Fy, Fz
