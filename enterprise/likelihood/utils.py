#utils.py
"""Likelihood utilities module containing
various useful functions.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np


def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    N = D + U*J*U.T
    :param Z:       The design matrix, array (n x m)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    :return zNz: Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc, 0]:Uinds[cc, 1], :]
                niblock = ni[Uinds[cc, 0]:Uinds[cc, 1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                zNz -= beta * np.outer(zn.T, zn)

    return zNz


def python_block_shermor_2D2(Z, X, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiX
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    N = D + U*J*U.T
    :param Z:       The design matrix, array (n x m)
    :param X:       The second design matrix, array (n x l)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    :return zNx: Z.T * N^-1 * X
    """
    ni = 1.0 / Nvec
    zNx = np.dot(Z.T*ni, X)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc, 0]:Uinds[cc, 1], :]
                Xblock = X[Uinds[cc, 0]:Uinds[cc, 1], :]
                niblock = ni[Uinds[cc, 0]:Uinds[cc, 1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                zNx -= beta * np.outer(zn.T, xn)

    return zNx


def python_block_shermor_0D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter.
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    :return Nx: N^{-1} * r
    """

    ni = 1/Nvec
    Nx = r/Nvec
    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc, 0]:Uinds[cc, 1]]
                niblock = ni[Uinds[cc, 0]:Uinds[cc, 1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                tmp = beta * np.dot(niblock, rblock) * niblock
                Nx[Uinds[cc, 0]:Uinds[cc, 1]] -= tmp

    return Nx


def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter.
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    N = D + U*J*U.T
    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    return Jldet, xNx: r.T * N^-1 * r, log(det(N))
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc, 0]:Uinds[cc, 1]]
                niblock = ni[Uinds[cc, 0]:Uinds[cc, 1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                xNx -= beta * np.dot(rblock, niblock)**2
                Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx


def quantize_fast(times, dt=1.0, calci=False):
    """ Adapted from libstempo: produce the quantisation matrix fast """

    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')

    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1

    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def quantize_split(times, flags, dt=1.0, calci=False):
    """
    As quantize_fast, but now split the blocks per backend. Note: for
    efficiency, this function assumes that the TOAs have been sorted by
    argsortTOAs. This is _NOT_ checked.
    """
    isort = np.arange(len(times))

    bucket_ref = [times[isort[0]]]
    bucket_flag = [flags[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt and flags[i] == bucket_flag[-1]:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_flag.append(flags[i])
            bucket_ind.append([i])

    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')

    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1

    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def argsortTOAs(toas, flags, which=None, dt=1.0):
    """
    Return the sort, and the inverse sort permutations of the TOAs, for the
    requested type of sorting
    NOTE: This one is _not_ optimized for efficiency yet
    (but is done only once)
    :param toas:    The toas that are to be sorted
    :param flags:   The flags that belong to each TOA
                    (indicates sys/backend)
    :param which:   Which type of sorting we will use
                    (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks,
                    default [10 secs]

    :return:    perm, perminv       (sorting permutation, and inverse)
    """

    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        isort = np.argsort(toas, kind='mergesort')
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)

                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Sort the indices of this epoch and backend
                        # We need mergesort here, because it is stable
                        # (A stable sort keeps items with the same key in the
                        # same relative order. )
                        episort = np.argsort(flagmask[colmask],
                                             kind='mergesort')
                        isort[colmask] = isort[colmask][episort]
                else:
                    # Only one element, always ok
                    pass

        # Now that we have a correct permutation, also construct the inverse
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    else:
        isort, iisort = np.arange(len(toas)), np.arange(len(toas))

    return isort, iisort


def checkTOAsort(toas, flags, which=None, dt=1.0):
    """
    Check whether the TOAs are indeed sorted as they should be according to the
    definition in argsortTOAs
    :param toas:    The toas that are supposed to be already sorted
    :param flags:   The flags that belong to each TOA
                    (indicates sys/backend)
    :param which:   Which type of sorting we will check
                    (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks,
                    default [10 secs]

    :return:    True/False
    """
    rv = True
    if which is None:
        isort = slice(None, None, None)
        #iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        if not np.all(isort == np.arange(len(isort))):
            rv = False
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        #isort = np.argsort(toas, kind='mergesort')
        isort = np.arange(len(toas))
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)

                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Keys are not sorted for this epoch/flag
                        rv = False
                else:
                    # Only one element, always ok
                    pass
    else:
        pass

    return rv


def checkquant(U, flags, uflagvals=None):
    """
    Check the quantization matrix for consistency with the flags
    :param U:           quantization matrix
    :param flags:       the flags of the TOAs
    :param uflagvals:   subset of flags that are not ignored
    :return:            True/False, whether or not consistent
    The quantization matrix is checked for three kinds of consistency:
    - Every quantization epoch has more than one observation
    - No quantization epoch has no observations
    - Only one flag is allowed per epoch
    """
    if uflagvals is None:
        uflagvals = list(set(flags))

    rv = True
    collisioncheck = np.zeros((U.shape[1], len(uflagvals)), dtype=np.int)
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)

        Umat = U[flagmask, :]

        simepoch = np.sum(Umat, axis=0)
        if np.all(simepoch <= 1) and not np.all(simepoch == 0):
            rv = False

        collisioncheck[:, ii] = simepoch

        # Check continuity of the columns
        for cc, col in enumerate(Umat.T):
            if np.sum(col > 2):
                # More than one TOA for this flag/epoch
                epinds = np.flatnonzero(col)
                if len(epinds) != epinds[-1] - epinds[0] + 1:
                    rv = False
                    print("WARNING: checkquant found non-continuous blocks")

    epochflags = np.sum(collisioncheck > 0, axis=1)

    if np.any(epochflags > 1):
        rv = False
        print("WARNING: checkquant found multiple backends for an epoch")

    if np.any(epochflags < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (eflags)")

    obsum = np.sum(U, axis=0)
    if np.any(obsum < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (all)")

    return rv


def quant2ind(U):
    """
    Convert the quantization matrix to an indices matrix
    for fast use in the jitter likelihoods.
    This function assumes that the TOAs have been properly
    sorted according to the proper function argsortTOAs above.
    Checks on the continuity of U are not performed.

    :param U:       quantization matrix

    :return:        Index (basic slicing) version of the quantization matrix

    """
    inds = np.zeros((U.shape[1], 2), dtype=np.int)
    for cc, col in enumerate(U.T):
        epinds = np.flatnonzero(col)
        inds[cc, 0] = epinds[0]
        inds[cc, 1] = epinds[-1] + 1

    return inds


def quantreduce(U, eat, flags, calci=False):
    """
    Reduce the quantization matrix by removing the observing epochs that do not
    require any jitter parameters.
    :param U:       quantization matrix
    :param eat:     Epoch-averaged toas
    :param flags:   the flags of the TOAs
    :param calci:   Calculate pseudo-inverse yes/no
    :return     newU, jflags (flags that need jitter)
    """
    uflagvals = list(set(flags))
    incepoch = np.zeros(U.shape[1], dtype=np.bool)
    jflags = []
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)

        Umat = U[flagmask, :]
        ecnt = np.sum(Umat, axis=0)
        incepoch = np.logical_or(incepoch, ecnt > 1)

        if np.any(ecnt > 1):
            jflags.append(flagval)

    Un = U[:, incepoch]
    eatn = eat[incepoch]

    if calci:
        Ui = ((1.0/np.sum(Un, axis=0)) * Un).T
        rv = (Un, Ui, eatn, jflags)
    else:
        rv = (Un, eatn, jflags)

    return rv
