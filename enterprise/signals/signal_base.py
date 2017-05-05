# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import six
import scipy
from sksparse.cholmod import cholesky

import enterprise.signals.utils as util
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

    def __init__(self, psr):
        self._psr = psr

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


def SignalCollection(metasignals):
    """Class factory for ``SignalCollection`` objects."""
    @six.add_metaclass(MetaCollection)
    class SignalCollection(object):
        _metasignals = metasignals

        def __init__(self, psr):
            self._psr = psr

            # instantiate all the signals with a pulsar
            self._signals = [metasignal(psr) for metasignal
                             in self._metasignals]

            self._cbasis_bool = False
            self._cbasis = {}

        #def __add__(self,other):
        #    return PTA([self,other])

        @property
        def params(self):

            # no duplicates, but expensive, so a candidate for memoization
            ret = []
            for signal in self._signals:
                for param in signal.params:
                    if param not in ret:
                        ret.append(param)

            return ret

        # TODO: use decorator for this
        def get_common_basis_mappings(self, params):
            if not self._cbasis_bool:
                self._cbasis_bool = True
                self._cbasis = util.get_independent_columns(
                    self.get_basis(params))
            return self._cbasis

        # there may be a smarter way to write these...

        def get_ndiag(self, params):
            ndiags = [signal.get_ndiag(params) for signal in self._signals]
            return sum(ndiag for ndiag in ndiags if ndiag is not None)

        def get_delay(self, params):
            delays = [signal.get_delay(params) for signal in self._signals]
            return sum(delay for delay in delays if delay is not None)

        def get_basis(self, params=None):
            Fmats = [signal.get_basis(params) for signal in self._signals]

            # TODO: there is probably a cleaner way to do this
            F = np.hstack(Fmat for Fmat in Fmats if Fmat is not None)
            imap = self.get_common_basis_mappings(params)
            idx = np.array([ii for ii in range(F.shape[1]) if ii
                            not in sum(imap.values(), [])])
            return F[:, idx]

        def get_phiinv(self, params):
            Phiinvs = [signal.get_phiinv(params) for signal in self._signals]
            phiinv = np.hstack(Phiinv for Phiinv in Phiinvs
                               if Phiinv is not None)
            imap = self.get_common_basis_mappings(params)
            idx = np.array([ii for ii in range(phiinv.shape[0]) if ii
                            not in sum(imap.values(), [])])
            for key, vals in imap.items():
                for v in vals:
                    phiinv[key] += phiinv[v]
            return phiinv[idx]

        def get_phi(self, params):
            Phivecs = [signal.get_phi(params) for signal in self._signals]
            phi = np.hstack(Phivec for Phivec in Phivecs if Phivec is not None)
            imap = self.get_common_basis_mappings(params)
            idx = np.array([ii for ii in range(phi.shape[0]) if ii
                            not in sum(imap.values(), [])])
            for key, vals in imap.items():
                for v in vals:
                    phi[key] += phi[v]
            return phi[idx]

    return SignalCollection


def Function(func, **kwargs):
    """Class factory for generic function calls."""
    class Function(object):
        def __init__(self, prefix):
            self._params = {kw: arg(prefix + '_' + kw)
                            for kw,arg in kwargs.items()}

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
