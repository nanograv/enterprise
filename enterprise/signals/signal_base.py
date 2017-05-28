# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import collections
import itertools
import logging

import six

import numpy as np
import scipy
import scipy.sparse as sps
import scipy.linalg as sl

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


class MarginalizedLogLikelihood(object):
    def __init__(self,pta):
        self.pta = pta

    # this can and should be much cleaner
    def __call__(self, xs):
        # map parameter vector if needed
        params = xs if isinstance(xs,dict) else self.pta.map_params(xs)

        # these are all lists in pulsar order
        # except for phiinv which may be a big matrix
        Nvecs = self.pta.get_ndiag(params)
        Ts = self.pta.get_basis(params)
        phiinvs = self.pta.get_phiinv(params,logdet=True)
        residuals = self.pta.get_residuals()

        loglike = 0.0

        ds, TNTs = [], []
        for T, Nvec, residual in zip(Ts, Nvecs, residuals):
            # get auxiliaries
            d = np.dot(T.T, Nvec.solve(residual))
            NT, logdet_N = Nvec.solve(T, logdet=True)
            TNT = np.dot(T.T, NT)

            # triple product in likelihood function
            rNr = np.dot(residual, Nvec.solve(residual))

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)

            # save d and TNT for use below
            ds.append(d)
            TNTs.append(TNT)

        # red noise piece
        if self.pta._commonsignals:
            phiinv, logdet_phi = phiinvs

            # note: modifies phiinv in place
            Sigma = phiinv + sps.block_diag(TNTs,'csc')
            d = np.concatenate(ds)

            cf = cholesky(Sigma)
            expval = cf(d)

            logdet_sigma = cf.logdet()

            loglike += 0.5*(np.dot(d, expval) - logdet_sigma - logdet_phi)
        else:
            for d, TNT, (phiinv, logdet_phi) in zip(ds, TNTs, phiinvs):
                Sigma = TNT + np.diag(phiinv)

                cf = sl.cho_factor(Sigma)
                expval = sl.cho_solve(cf, d)

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))

                loglike += 0.5*(np.dot(d, expval) - logdet_sigma - logdet_phi)

        return loglike


class PTA(object):
    def __init__(self, init, lnlikelihood=MarginalizedLogLikelihood):
        if isinstance(init, collections.Sequence):
            self._signalcollections = list(init)
        else:
            self._signalcollections = [init]

        self.lnlikelihood = lnlikelihood

    def __add__(self, other):
        if hasattr(other, '_signalcollections'):
            return PTA(self._signalcollections+other._signalcollections,
                       lnlikelihood=self.lnlikelihood)
        else:
            return PTA(self._signalcollections+[other],
                       lnlikelihood=self.lnlikelihood)

    @property
    def params(self):
        return sorted({par for signalcollection in self._signalcollections for
                       par in signalcollection.params},
                      key=lambda par: par.name)

    def get_residuals(self):
        return [signalcollection._residuals
                for signalcollection in self._signalcollections]

    def get_ndiag(self, params):
        return [signalcollection.get_ndiag(params)
                for signalcollection in self._signalcollections]

    def get_basis(self, params=None):
        return [signalcollection.get_basis(params) for
                signalcollection in self._signalcollections]

    @property
    def _lnlikelihood(self):
        # instantiate on first use
        if not hasattr(self, '_lnlike'):
            self._lnlike = self.lnlikelihood(self)

        return self._lnlike

    def get_lnlikelihood(self, params):
        return self._lnlikelihood(params)

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

    def get_phiinv(self, params, logdet=False, method='cliques'):
        if method == 'cliques':
            return self.get_phiinv_byfreq_cliques(params, logdet)
        elif method == 'partition':
            return self.get_phiinv_byfreq_partition(params, logdet)
        elif method == 'sparse':
            return self.get_phiinv_sparse(params, logdet)
        else:
            raise NotImplementedError

    def get_phiinv_sparse(self, params, logdet=False):
        phi = self.get_phi(params)

        if isinstance(phi, list):
            if logdet:
                return [None if phivec is None
                        else (1/phivec, np.sum(np.log(phivec)))
                        for phivec in phi]
            else:
                return [None if phivec is None else 1/phivec for phivec in phi]
        else:
            phisparse = sps.csc_matrix(phi)
            cf = cholesky(phisparse)

            if logdet:
                return (cf.inv(), cf.logdet())
            else:
                return cf.inv()

    def get_phiinv_byfreq_partition(self, params, logdet=False):
        phivecs = [signalcollection.get_phi(params) for
                   signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            # would be easier if get_phi would return an empty array
            phidiag = np.concatenate([phivec for phivec in phivecs
                                      if phivec is not None])
            slices = self._get_slices(phivecs)

            if logdet:
                ld = np.sum(np.log(phidiag))

            phiinv = np.diag(1.0/phidiag)

            # assume no superposition between common signals
            for csclass, csdict in self._commonsignals.items():
                invert = None

                for i, (cs1, csc1) in enumerate(csdict.items()):
                    for j, (cs2, csc2) in enumerate(csdict.items()):
                        if j <= i: continue

                        # hoping they're all the same...
                        crossdiag = csclass.get_phicross(cs1, cs2, params)

                        if invert is None:
                            invert = np.zeros((len(crossdiag),len(csdict),len(csdict)),'d')

                        invert[:,i,j] += crossdiag
                        invert[:,j,i] += crossdiag

                    invert[:,i,i] += phidiag[slices[csc1]][csc1._idx[cs1]]
                    
                    if logdet:
                        ld -= np.sum(np.log(phidiag[slices[csc1]][csc1._idx[cs1]]))

                for k in range(len(crossdiag)):
                    if logdet:
                        ld += np.linalg.slogdet(invert[k,:,:])[1]

                    invert[k,:,:] = np.linalg.inv(invert[k,:,:])    

                for i, (cs1, csc1) in enumerate(csdict.items()):
                    for j, (cs2, csc2) in enumerate(csdict.items()):
                        if j < i: continue

                        block1, idx1 = slices[csc1], csc1._idx[cs1]
                        block2, idx2 = slices[csc2], csc2._idx[cs2]

                        phiinv[block1,block2][idx1,idx2] = invert[:,i,j]
                        phiinv[block2,block1][idx2,idx1] = invert[:,i,j]

            if logdet:
                return phiinv, ld
            else:
                return phiinv
        else:
            raise NotImplementedError

    def get_phiinv_byfreq_cliques(self, params, logdet=False, cholesky=False):
        phi = self.get_phi(params, cliques=True)

        if isinstance(phi, list):
            if logdet:
                return [None if phivec is None
                        else (1/phivec, np.sum(np.log(phivec)))
                        for phivec in phi]
            else:
                return [None if phivec is None else 1/phivec for phivec in phi]
        else:
            ld = 1.0

            # first invert all the cliques
            for clcount in range(self._clcount):
                idx = (self._cliques == clcount)

                if np.any(idx):
                    idx2 = np.ix_(idx,idx)

                    if cholesky:
                        cf = sl.cho_factor(phi[idx2])
                        
                        if logdet:
                            ld += 2.0*np.sum(np.log(np.diag(cf[0])))
                        
                        phi[idx2] = sl.cho_solve(cf,np.identity(cf[0].shape[0]))
                    else:
                        phi2 = phi[idx2]

                        if logdet:
                            ld += np.linalg.slogdet(phi2)[1]

                        phi[idx2] = np.linalg.inv(phi2)

            # then do the pure diagonal terms
            idx = (self._cliques == -1)

            if logdet:
                ld += np.sum(np.log(phi[idx,idx]))

            phi[idx,idx] = 1.0/phi[idx,idx]

            return (phi, ld) if logdet else phi

    # sort matrix indices by presence of non-diagonal elements
    # for each value in self._cliques, the indices with that value form
    # an independent submatrix that can be inverted separately   
    def _resetcliques(self,phidiag):
        self._cliques = -1 * np.ones_like(phidiag)
        self._clcount = 0

    def _setcliques(self,slices,csdict):
        idxmatrix = np.array([csc._idx[cs] for cs, csc in csdict.items()]).T
        idxmatrix = idxmatrix + np.array([slices[csc].start for cs, csc in csdict.items()])

        for idxs in idxmatrix:
            allidx = set(self._cliques[idxs])
            maxidx = max(allidx)

            if maxidx == -1:
                self._cliques[idxs] = self._clcount

                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques,allidx)] = self._clcount

                self._clcount = self._clcount + 1
            else:
                self._cliques[idxs] = maxidx
                
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques,allidx)] = maxidx

    def get_phi(self, params, cliques=False):
        phivecs = [signalcollection.get_phi(params) for
                   signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            # would be easier if get_phi would return an empty array
            phidiag = np.concatenate([phivec for phivec in phivecs
                                      if phivec is not None])
            slices = self._get_slices(phivecs)

            phi = np.diag(phidiag)

            if cliques:
                self._resetcliques(phidiag)

            # iterate over all common signal classes
            for csclass, csdict in self._commonsignals.items():
                # first figure out which indices are used in this common signal
                if cliques:
                    self._setcliques(slices,csdict)

                # now iterate over all pairs of common signal instances
                pairs = itertools.combinations(csdict.items(),2)

                for (cs1, csc1), (cs2, csc2) in pairs:
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    phi[block1,block2][idx1,idx2] += crossdiag
                    phi[block2,block1][idx2,idx1] += crossdiag

            return phi
        else:
            return phivecs

    def map_params(self, xs):
        return {par.name: x for par, x in zip(self.params, xs)}

    def get_lnprior(self, xs):
        # map parameter vector if needed
        params = xs if isinstance(xs,dict) else self.map_params(xs)

        return np.sum(p.get_logpdf(params[p.name]) for p in self.params)


def SignalCollection(metasignals):
    """Class factory for ``SignalCollection`` objects."""

    @six.add_metaclass(MetaCollection)
    class SignalCollection(object):
        _metasignals = metasignals

        def __init__(self, psr):
            # instantiate all the signals with a pulsar
            self._signals = [metasignal(psr) for metasignal
                             in self._metasignals]

            self._residuals = psr.residuals

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
