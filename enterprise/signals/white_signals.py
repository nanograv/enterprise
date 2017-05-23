# white_signals.py
"""Contains class factories for white noise signals. White noise signals are
defined as the class of signals that only modifies the white noise matrix `N`.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import scipy.sparse

from enterprise.signals import parameter
import enterprise.signals.signal_base as base
from enterprise.signals import utils
from enterprise.signals import selections
from enterprise.signals.selections import Selection


def MeasurementNoise(efac=parameter.Uniform(0.5,1.5),
                     selection=Selection(selections.no_selection)):
    """Class factory for EFAC type measurement noise."""

    class MeasurementNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'efac'

        def __init__(self, psr):

            sel = selection(psr)
            self._params, self._ndiag = sel('efac', efac, psr.toaerrs**2)

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [self.get(p, params)**2*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return MeasurementNoise


def EquadNoise(log10_equad=parameter.Uniform(-10,-5),
               selection=Selection(selections.no_selection)):
    """Class factory for EQUAD type measurement noise."""

    class EquadNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'equad'

        def __init__(self,psr):

            sel = selection(psr)
            self._params, self._ndiag = sel('log10_equad', log10_equad,
                                            np.ones_like(psr.toaerrs))

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [10**(2*self.get(p, params))*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return EquadNoise


def EcorrKernelNoise(log10_ecorr=parameter.Uniform(-10, -5),
                     selection=Selection(selections.no_selection),
                     method='sherman-morrison'):
    """Class factory for ECORR type noise.

    :param log10_ecorr: ``Parameter`` type for log10 or ecorr parameter.
    :param selection:
        ``Selection`` object specifying masks for backends, time segments, etc.
    :param method: Method for computing noise covariance matrix.
        Options include `sherman-morrison`, `sparse`, and `block`

    :return: ``EcorrKernelNoise`` class.

    ECORR is a noise signal that is used for data with multi-channel TOAs
    that are nearly simultaneous in time. It is a white noise signal that
    is uncorrelated epoch to epoch but completely correlated for TOAs in a
    given observing epoch.

    Mathematically, ECORR can be described by the covariance matrix

    .. math:: C_{ecorr} = UJU^T,

    where :math:`U` is a quantization matrix that maps TOAs to
    their respective epochs and :math:`J` is a diagonal matrix of the
    variance of the epoch to epoch fluctuations.

    For this implementation we use this covariance matrix as part of the
    white noise covariance matrix :math:`N`. It can be seen from above that
    this covariance is block diagonal, thus allowing us to exploit special
    methods to make matrix manipulations easier.

    In this signal implementation we offer three methods of performing these
    matrix operations:

    sherman-morrison
        Uses the `Sherman-Morrison`_ forumla to compute the matrix
        inverse and other matrix operations. **Note:** This method can only
        be used for covariances that can be constructed by the outer product
        of two vectors, :math:`uv^T`.

    sparse
        Uses `Scipy Sparse`_ matrices to construct the block diagonal
        covariance matrix and perform matrix operations.

    block
        Uses a custom scheme that uses the individual blocks from the block
        diagonal matrix to perform fast matrix inverse and other solve
        operations.

    .. note:: The sherman-morrison method is the fastest, followed by the block
        and then sparse methods, however; the block and sparse methods are more
        general and should be used if sub-classing this signal for more complicated
        blocks.

    .. _Sherman-Morrison: https://en.wikipedia.org/wiki/Sherman-Morrison_formula
    .. _Scipy Sparse: https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html
    .. # noqa E501

    """

    if method not in ['sherman-morrison', 'block', 'sparse']:
        msg = 'EcorrKernelNoise does not support method: {}'.format(method)
        raise TypeError(msg)

    class EcorrKernelNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'ecorr_' + method

        def __init__(self, psr):

            sel = selection(psr)
            self._params, self._masks = sel('log10_ecorr', log10_ecorr)
            keys = list(sorted(self._masks.keys()))
            masks = [self._masks[key] for key in keys]

            Umats = []
            for key, mask in zip(keys, masks):
                Umats.append(utils.create_quantization_matrix(
                    psr.toas[mask], nmin=1))

            nepoch = np.sum(U.shape[1] for U in Umats)
            self._F = np.zeros((len(psr.toas), nepoch))
            self._slices = {}
            netot = 0
            for ct, (key, mask) in enumerate(zip(keys, masks)):
                nn = Umats[ct].shape[1]
                self._F[mask, netot:nn+netot] = Umats[ct]
                self._slices.update({key: utils.quant2ind(
                    self._F[:,netot:nn+netot])})
                netot += nn

            # initialize sparse matrix
            self._setup(psr)

        def get_ndiag(self, params):
            if method == 'sherman-morrison':
                return self._get_ndiag_sherman_morrison(params)
            elif method == 'sparse':
                return self._get_ndiag_sparse(params)
            elif method == 'block':
                return self._get_ndiag_block(params)

        def _setup(self, psr):
            if method == 'sparse':
                self._setup_sparse(psr)

        def _setup_sparse(self, psr):
            Ns = scipy.sparse.csc_matrix((len(psr.toas), len(psr.toas)))
            for key, slices in self._slices.items():
                for slc in slices:
                    if slc.stop - slc.start > 1:
                        Ns[slc, slc] = 1.0
            self._Ns = base.csc_matrix_alt(Ns)

        def _get_ndiag_sparse(self, params):
            for p in self._params:
                for slc in self._slices[p]:
                    if slc.stop - slc.start > 1:
                        self._Ns[slc, slc] = 10**(2*self.get(p, params))
            return self._Ns

        def _get_ndiag_sherman_morrison(self, params):
            slices, jvec = self._get_jvecs(params)
            return base.ShermanMorrison(jvec, slices)

        def _get_ndiag_block(self, params):
            slices, jvec = self._get_jvecs(params)
            blocks = []
            for jv, slc in zip(jvec, slices):
                nb = slc.stop - slc.start
                blocks.append(np.ones((nb, nb))*jv)
            return base.BlockMatrix(blocks, slices)

        def _get_jvecs(self, params):
            slices = sum([self._slices[key] for key in
                          sorted(self._slices.keys())], [])
            jvec = np.concatenate(
                [np.ones(len(self._slices[key]))*10**(2*self.get(key, params))
                 for key in sorted(self._slices.keys())])
            return (slices, jvec)

    return EcorrKernelNoise
