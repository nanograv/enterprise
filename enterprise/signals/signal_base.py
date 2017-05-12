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
import scipy
from sksparse.cholmod import cholesky

from enterprise.signals.parameter import ConstantParameter


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

    def get_basis(self, params=None):
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

        # this could be put in utils.py if desired
        def _combine_basis_columns(self, signals):
            """Given a set of Signal objects, each of which may return an
            Fmat (through get_basis()), combine the unique columns into a
            single Fmat, dropping duplicates, and also return a
            dict (indexed by signal) of integer arrays that map individual
            Fmat columns to the combined Fmat."""

            idx, Fmatlist = {}, []

            for signal in signals:
                Fmat = signal.get_basis()

                if Fmat is not None:
                    idx[signal] = []

                    for i, column in enumerate(Fmat.T):
                        for j, savedcolumn in enumerate(Fmatlist):
                            if np.allclose(column, savedcolumn, rtol=1e-15):
                                idx[signal].append(j)
                                break
                        else:
                            idx[signal].append(len(Fmatlist))
                            Fmatlist.append(column)

            return idx, np.array(Fmatlist).T

        # goofy way to cache _idx and _Fmat
        def __getattr__(self, par):
            if par in ('_idx', '_Fmat'):
                self._idx, self._Fmat = self._combine_basis_columns(
                    self._signals)

                return getattr(self,par)
            else:
                raise AttributeError("{} object has no attribute {}".format(
                    self.__class__,par))

        # note that currently we don't support a params-dependent basis;
        # for that, we'll need Signal.get_basis (or an associated function)
        # to somehow return the set of parameters that matter to the basis,
        # which could be empty
        def get_basis(self, params=None):
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


def Function(func, **kwargs):
    """Class factory for generic function calls."""
    class Function(object):
        def __init__(self, prefix, postfix=''):
            self._params = {kw: arg('_'.join([prefix, kw, postfix]))
                            if postfix else arg('_'.join([prefix, kw]))
                            for kw, arg in kwargs.items()}

        def get(self, parname, params={}):
            try:
                return self._params[parname].value
            except AttributeError:
                return params[parname]

        # params could also be a standard argument here,
        # but by defining it as ** we allow multiple positional arguments
        def __call__(self, *args, **params):
            pardict = {}
            for kw, par in self._params.items():
                if par.name in params:
                    pardict[kw] = params[par.name]
                elif hasattr(par, 'value'):
                    pardict[kw] = par.value

            return func(*args, **pardict)

        @property
        def params(self):
            return [par for par in self._params.values() if not
                    isinstance(par,ConstantParameter)]

    return Function


class csc_matrix_alt(scipy.sparse.csc_matrix):
    """Sub-class of ``scipy.sparse.csc_matrix`` with custom ``add`` and
    ``solve`` methods.
    """

    def _add_diag(self, other):
        other_diag = scipy.sparse.dia_matrix(
            (other, np.array([0])),
            shape=(other.shape[0], other.shape[0]))
        return self._binopt(other_diag, '_plus_')

    def __add__(self, other):

        if isinstance(other, np.ndarray) and other.ndim == 1:
            return self._add_diag(other)
        else:
            return super(csc_matrix_alt, self).__add__(other)

    def solve(self, other, logdet=False):
        cf = cholesky(self)
        ret = (cf(other), cf.logdet()) if logdet else cf(other)
        return ret


class ndarray_alt(np.ndarray):
    """Sub-class of ``np.ndarray`` with custom ``solve`` method."""

    def __new__(cls, inputarr):
        obj = np.asarray(inputarr).view(cls)
        return obj

    def solve(self, other, logdet=False):
        if other.ndim == 1:
            No = np.array(other / self)
        elif other.ndim == 2:
            No = np.array(other / self[:,None])

        ret = (No, float(np.sum(np.log(self)))) if logdet else No
        return ret
