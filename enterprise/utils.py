#utils.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np


def createfourierdesignmatrix(t, nmodes, freq=False, Tspan=None,
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
    :return: ranphase: Phase offsets applied to basis
    functions of each frequency
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
    ct = 0
    for ii in range(0, 2 * nmodes - 1, 2):

        if pshift:
            F[:, ii] = np.cos(2 * np.pi * f[ct] * t + ranphase[ct])
            F[:, ii + 1] = np.sin(2 * np.pi * f[ct] * t + ranphase[ct])
        elif not pshift:
            F[:, ii] = np.cos(2 * np.pi * f[ct] * t)
            F[:, ii + 1] = np.sin(2 * np.pi * f[ct] * t)

        ct += 1

    if freq:
        return F, Ffreqs, ranphase
    else:
        return F, ranphase
