# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import collections
from itertools import combinations
import six
import scipy.sparse as sps
import scipy.linalg as sl
from sksparse.cholmod import cholesky

from enterprise.signals.parameter import ConstantParameter, Parameter
from enterprise.signals.selections import selection_func


class MetaSignal(type):
    """Metaclass for Signals. Allows addition of ``Signal`` classes."""

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection([self, other])
        elif isinstance(other, MetaCollection):
            return SignalCollection([self]+other._metasignals)
        else:
            raise TypeError


class MetaCollection(type):
    """Metaclass for Signal collections. Allows addition of
    ``SignalCollection`` classes.
    """

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection(self._metasignals+[other])
        elif isinstance(other, MetaCollection):
            return SignalCollection(self._metasignals+other._metasignals)
        else:
            raise TypeError


@six.add_metaclass(MetaSignal)
class Signal(object):
    """Base class for Signal objects."""

    @property
    def params(self):
        # return only nonconstant parameters
        return [par for par in self._params.values() if not
                isinstance(par, ConstantParameter)]

    def get(self, parname, params={}):
        try:
            return params[self._params[parname].name]
        except KeyError:
            return self._params[parname].value

    def get_ndiag(self, params):
        """Returns the diagonal of the white noise vector `N`.

        This method also supports block diagaonal sparse matrices.
        """
        return None

    def get_delay(self, params):
        """Returns the waveform of a deterministic signal."""
        return None

    def get_basis(self, params=None):
        """Returns the basis array of shape N_toa x N_basis."""
        return None

    def get_phi(self, params):
        """Returns a diagonal covaraince matrix of the basis amplitudes."""
        return None

    def get_phiinv(self, params):
        """Returns inverse of the covaraince of basis amplitudes."""
        return None


class CommonSignal(Signal):
    """Base class for CommonSignal objects."""

    def get_phiinv(self, params):
        msg = "You probably shouldn't be calling get_phiinv() "
        msg += "on a common red-noise signal."
        raise RuntimeError(msg)

    @classmethod
    def get_phicross(cls, signal1, signal2, params):
        return None


class PTA(object):
    def __init__(self, init):
        if isinstance(init, collections.Sequence):
            self._signalcollections = list(init)
        else:
            self._signalcollections = [init]

    def __add__(self, other):
        if hasattr(other, '_signalcollections'):
            return PTA(self._signalcollections+other._signalcollections)
        else:
            return PTA(self._signalcollections+[other])

    @property
    def params(self):
        return sorted({par for signalcollection in self._signalcollections for
                       par in signalcollection.params},
                      key=lambda par: par.name)

    def get_basis(self, params={}):
        return [signalcollection.get_basis(params) for
                signalcollection in self._signalcollections]

    def get_phiinv(self, params):
        phi = self.get_phi(params)

        if isinstance(phi, list):
            return [None if phivec is None else 1/phivec for phivec in phi]
        else:
            # TODO: replace with suitable sparse matrix definition
            return np.linalg.inv(phi)

    @property
    def _commonsignals(self):
        # cache the computation if we don't have it yet
        if not hasattr(self, '_cs'):
            commonsignals = collections.defaultdict(collections.OrderedDict)

            for signalcollection in self._signalcollections:
                # TODO: need a better signal that a
                # signalcollection provides a basis
                if signalcollection._Fmat is not None:
                    for signal in signalcollection._signals:
                        if isinstance(signal, CommonSignal):
                            commonsignals[signal.__class__][signal] = \
                                signalcollection

            # drop common signals that appear only once
            self._cs = {csclass: csdict for csclass, csdict in
                        commonsignals.items() if len(csdict) > 1}

        return self._cs

    def _get_slices(self, phivecs):
        ret, offset = {}, 0
        for sc, phivec in zip(self._signalcollections, phivecs):
            stop = 0 if phivec is None else len(phivec)
            ret[sc] = slice(offset, offset+stop)
            offset = ret[sc].stop

        return ret

    def get_phi(self, params):
        phivecs = [signalcollection.get_phi(params) for
                   signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            # would be easier if get_phi would return an empty array
            phidiag = np.concatenate([phivec for phivec in phivecs
                                      if phivec is not None])
            slices = self._get_slices(phivecs)

            # TODO: replace with suitable sparse matrix definition
            phi = np.diag(phidiag)

            # iterate over all common signal classes
            for csclass, csdict in self._commonsignals.items():
                # iterate over all pairs of common signal instances
                for (cs1, csc1), (cs2, csc2) in combinations(csdict.items(),2):
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    phi[block1,block2][idx1,idx2] += crossdiag
                    phi[block2,block1][idx2,idx1] += crossdiag

            return phi
        else:
            return phivecs


def SignalCollection(metasignals):
    """Class factory for ``SignalCollection`` objects."""

    @six.add_metaclass(MetaCollection)
    class SignalCollection(object):
        _metasignals = metasignals

        def __init__(self, psr):

            # instantiate all the signals with a pulsar
            self._signals = [metasignal(psr) for metasignal
                             in self._metasignals]

        def __add__(self, other):
            return PTA([self, other])

        # a candidate for memoization
        @property
        def params(self):
            return sorted({param for signal in self._signals for param
                           in signal.params}, key=lambda par: par.name)

        # there may be a smarter way to write these...

        def get_ndiag(self, params):
            ndiags = [signal.get_ndiag(params) for signal in self._signals]
            return sum(ndiag for ndiag in ndiags if ndiag is not None)

        def get_delay(self, params):
            delays = [signal.get_delay(params) for signal in self._signals]
            return sum(delay for delay in delays if delay is not None)

        def _combine_basis_columns(self, signals):
            """Given a set of Signal objects, each of which may return an
            Fmat (through get_basis()), return a dict (indexed by signal)
            of integer arrays that map individual Fmat columns to the
            combined Fmat.

            Note: The Fmat returned here is simply meant to initialize the
            matrix to save computations when calling `get_basis` later.
            """

            idx, Fmatlist = {}, []
            cc = 0
            for signal in signals:
                Fmat = signal.get_basis()

                if Fmat is not None and not signal.basis_params:
                    idx[signal] = []

                    for i, column in enumerate(Fmat.T):
                        for j, savedcolumn in enumerate(Fmatlist):
                            if np.allclose(column,savedcolumn,rtol=1e-15):
                                idx[signal].append(j)
                                break
                        else:
                            idx[signal].append(cc)
                            Fmatlist.append(column)
                            cc += 1

                elif Fmat is not None and signal.basis_params:
                    nf = Fmat.shape[1]
                    idx[signal] = list(np.arange(cc, cc+nf))
                    cc += nf

            ncol = len(np.unique(sum(idx.values(), [])))
            nrow = len(Fmatlist[0])
            return idx, np.zeros((nrow, ncol))

        # goofy way to cache _idx
        def __getattr__(self, par):
            if par in ('_idx', '_Fmat'):
                self._idx, self._Fmat = self._combine_basis_columns(
                    self._signals)
                return getattr(self,par)
            else:
                raise AttributeError("{} object has no attribute {}".format(
                    self.__class__,par))

        def get_basis(self, params={}):
            for signal in self._signals:
                if signal in self._idx:
                    self._Fmat[:, self._idx[signal]] = signal.get_basis(params)
            return self._Fmat

        def get_phiinv(self, params):
            return 1.0/self.get_phi(params)

        def get_phi(self, params):
            phi = np.zeros(self._Fmat.shape[1],'d')
            for signal in self._signals:
                if signal in self._idx:
                    phi[self._idx[signal]] += signal.get_phi(params)
            return phi

        @property
        def basis_shape(self):
            return self._Fmat.shape

    return SignalCollection


def Function(func, name='', **func_kwargs):
    fname = name

    class Function(object):
        def __init__(self, name, psr=None):
            self._func = selection_func(func)
            self._psr = psr

            self._params = {}
            self._defaults = {}

            # divide keyword parameters into those that are Parameter classes,
            # Parameter instances (useful for global parameters),
            # and something else (which we will assume is a value)
            for kw, arg in func_kwargs.items():
                if isinstance(arg, type) and issubclass(
                        arg, (Parameter, ConstantParameter)):
                    # parameter name template
                    # pname_[signalname_][fname_]parname
                    pnames = [name, fname, kw]
                    par = arg('_'.join([n for n in pnames if n]))
                    self._params[kw] = par
                elif isinstance(arg, (Parameter, ConstantParameter)):
                    self._params[kw] = arg
                else:
                    self._defaults[kw] = arg

        def __call__(self, *args, **kwargs):
            # order of parameter resolution:
            # - parameter given in kwargs
            # - named sampling parameter in self._params, if given in params
            #   or if it has a value
            # - parameter given as constant in Function definition
            # - default value for keyword parameter in func definition

            # trick to get positional arguments before params kwarg
            params = kwargs.get('params',{})
            if 'params' in kwargs:
                del kwargs['params']

            for kw, arg in func_kwargs.items():
                if kw not in kwargs and kw in self._params:
                    par = self._params[kw]

                    if par.name in params:
                        kwargs[kw] = params[par.name]
                    elif hasattr(par, 'value'):
                        kwargs[kw] = par.value

            for kw, arg in self._defaults.items():
                if kw not in kwargs:
                    kwargs[kw] = arg

            if self._psr is not None and 'psr' not in kwargs:
                kwargs['psr'] = self._psr
            return self._func(*args, **kwargs)

        @property
        def params(self):
            # if we extract the ConstantParameter value above, we would not
            # need a special case here
            return [par for par in self._params.values() if not
                    isinstance(par, ConstantParameter)]

    return Function


def cache_call(attr, limit=10):
    """Cache function that allows for subsets of parameters to be keyed."""

    def cache_decorator(func):

        def wrapper(self, params):
            keys = getattr(self, attr)
            key = tuple([(key, params[key]) for key in keys if key in params])
            if key not in self._cache:
                self._cache_list.append(key)
                self._cache[key] = func(self, params)
                if len(self._cache_list) > limit:
                    del self._cache[self._cache_list.pop(0)]
                return self._cache[key]
        return wrapper

    return cache_decorator


class csc_matrix_alt(sps.csc_matrix):
    """Sub-class of ``scipy.sparse.csc_matrix`` with custom ``add`` and
    ``solve`` methods.
    """

    def _add_diag(self, other):
        other_diag = sps.dia_matrix(
            (other, np.array([0])),
            shape=(other.shape[0], other.shape[0]))
        return self._binopt(other_diag, '_plus_')

    def __add__(self, other):

        if isinstance(other, (np.ndarray, ndarray_alt)) and other.ndim == 1:
            return self._add_diag(other)
        else:
            return super(csc_matrix_alt, self).__add__(other)

    # hacky way to fix adding ndarray on left
    def __radd__(self, other):
        if isinstance(other, (np.ndarray, ndarray_alt)) or other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def solve(self, other, left_array=None, logdet=False):
        cf = cholesky(self)
        mult = cf(other)
        if left_array is not None:
            mult = np.dot(left_array.T, mult)
        ret = (mult, cf.logdet()) if logdet else mult
        return ret


class ndarray_alt(np.ndarray):
    """Sub-class of ``np.ndarray`` with custom ``solve`` method."""

    def __new__(cls, inputarr):
        obj = np.asarray(inputarr).view(cls)
        return obj

    def __add__(self, other):
        try:
            ret = super(ndarray_alt, self).__add__(other)
        except TypeError:
            ret = other + self
        return ret

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            mult = np.array(other / self)
        elif other.ndim == 2:
            mult = np.array(other / self[:,None])
        if left_array is not None:
            mult = np.dot(left_array.T, mult)

        ret = (mult, float(np.sum(np.log(self)))) if logdet else mult
        return ret


class BlockMatrix(object):

    def __init__(self, blocks, slices, nvec=0):
        self._blocks = blocks
        self._slices = slices
        self._nvec = nvec

    def __add__(self, other):
        nvec = self._nvec + other
        return BlockMatrix(self._blocks, self._slices, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_ZNX(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 1-d or 2-d arrays.
        """
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        if Z.ndim == 1:
            Z = Z.reshape(Z.shape[0], 1)

        n, m = Z.shape[1], X.shape[1]
        ZNX = np.zeros((n, m))
        for slc, block in zip(self._slices, self._blocks):
            Zblock = Z[slc, :]
            Xblock = X[slc, :]

            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block+np.diag(self._nvec[slc]))
                bx = sl.cho_solve(cf, Xblock)
            else:
                bx = Xblock / self._nvec[slc][:, None]
            ZNX += np.dot(Zblock.T, bx)
        return ZNX.squeeze() if len(ZNX) > 1 else float(ZNX)

    def _solve_NX(self, X):
        """Solves :math:`N^{-1}X`, where :math:`X`
        is a 1-d or 2-d array.
        """
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        m = X.shape[1]
        NX = np.zeros((len(self._nvec), m))
        for slc, block in zip(self._slices, self._blocks):
            Xblock = X[slc, :]
            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block+np.diag(self._nvec[slc]))
                NX[slc] = sl.cho_solve(cf, Xblock)
            else:
                NX[slc] = Xblock / self._nvec[slc][:, None]
        return NX.squeeze()

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logdet = 0
        for slc, block in zip(self._slices, self._blocks):
            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block+np.diag(self._nvec[slc]))
                logdet += np.sum(2*np.log(np.diag(cf[0])))
            else:
                logdet += np.sum(np.log(self._nvec[slc]))
        return logdet

    def solve(self, other, left_array=None, logdet=False):

        if other.ndim not in [1, 2]:
            raise TypeError
        if left_array is not None:
            if left_array.ndim not in [1, 2]:
                raise TypeError

        if left_array is not None:
            ret = self._solve_ZNX(other, left_array)
        else:
            ret = self._solve_NX(other)

        return (ret, self._get_logdet()) if logdet else ret


class ShermanMorrison(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        self._jvec = jvec
        self._slices = slices
        self._nvec = nvec

    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison(self._jvec, self._slices, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum('i->', niblock) + 1.0/jv)
                Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                ZNX -= beta * np.outer(zn.T, xn)
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logdet = np.einsum('i->', np.log(self._nvec))
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                logdet += np.log(jv) - np.log(beta)
        return logdet

    def solve(self, other, left_array=None, logdet=False):

        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise TypeError
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret
