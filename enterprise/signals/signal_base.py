# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import collections
import itertools
import inspect
import functools

import six

import numpy as np
import scipy.sparse as sps
import scipy.linalg as sl

from sksparse.cholmod import cholesky

from enterprise.signals.parameter import ConstantParameter, Parameter
from enterprise.signals.selections import selection_func

import logging
logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name+'_{}'.format(ii))
            else:
                ret.append(p.name)
        return ret

    def get(self, parname, params={}):
        try:
            return params[self._params[parname].name]
        except KeyError:
            return self._params[parname].value

    def set_default_params(self, params):
        """Set default parameters."""
        for kw, par in self._params.items():
            if par.name in params and isinstance(par, ConstantParameter):
                msg = 'Setting {} to {}'.format(par.name, params[par.name])
                logger.info(msg)
                self._params[kw].value = params[par.name]
            elif par.name not in params and isinstance(par, ConstantParameter):
                if par.value is not None:
                    msg = '{} not set! Check parameter dict.'.format(par.name)
                    logger.warning(msg)

    def get_ndiag(self, params):
        """Returns the diagonal of the white noise vector `N`.

        This method also supports block diagaonal sparse matrices.
        """
        return None

    def get_delay(self, params):
        """Returns the waveform of a deterministic signal."""
        return 0

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
    def __init__(self, pta):
        self.pta = pta

    def _make_sigma(self, TNTs, phiinv):
        return sps.block_diag(TNTs,'csc') + sps.csc_matrix(phiinv)

    def __call__(self, xs, phiinv_method='partition'):
        # map parameter vector if needed
        params = xs if isinstance(xs,dict) else self.pta.map_params(xs)

        # phiinvs will be a list or may be a big matrix if spatially
        # correlated signals
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=True,
                                      method=phiinv_method)

        # get -0.5 * (rNr + logdet_N) piece of likelihood
        loglike = -0.5 * np.sum([l for l in self.pta.get_rNr_logdet(params)])

        # red noise piece
        if self.pta._commonsignals:
            phiinv, logdet_phi = phiinvs

            Sigma = self._make_sigma(TNTs, phiinv)
            TNr = np.concatenate(TNrs)

            cf = cholesky(Sigma)
            expval = cf(TNr)

            logdet_sigma = cf.logdet()

            loglike += 0.5*(np.dot(TNr, expval) - logdet_sigma - logdet_phi)
        else:
            for TNr, TNT, (phiinv, logdet_phi) in zip(TNrs, TNTs, phiinvs):
                Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                except:
                    return -np.inf

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))

                loglike += 0.5*(np.dot(TNr, expval) -
                                logdet_sigma - logdet_phi)

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

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name+'_{}'.format(ii))
            else:
                ret.append(p.name)
        return ret

    def get_TNr(self, params):
        return [signalcollection.get_TNr(params) for signalcollection
                in self._signalcollections]

    def get_TNT(self, params):
        return [signalcollection.get_TNT(params) for signalcollection
                in self._signalcollections]

    def get_rNr_logdet(self, params):
        return [signalcollection.get_rNr_logdet(params) for signalcollection
                in self._signalcollections]

    def get_residuals(self):
        return [signalcollection._residuals
                for signalcollection in self._signalcollections]

    def get_ndiag(self, params={}):
        return [signalcollection.get_ndiag(params)
                for signalcollection in self._signalcollections]

    def get_delay(self, params={}):
        return [signalcollection.get_delay(params)
                for signalcollection in self._signalcollections]

    def set_default_params(self, params):
        for sc in self._signalcollections:
            sc.set_default_params(params)

    def get_basis(self, params={}):
        return [signalcollection.get_basis(params) for
                signalcollection in self._signalcollections]

    @property
    def _lnlikelihood(self):
        # instantiate on first use
        if not hasattr(self, '_lnlike'):
            self._lnlike = self.lnlikelihood(self)

        return self._lnlike

    def get_lnlikelihood(self, params, **kwargs):
        return self._lnlikelihood(params, **kwargs)

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

    # return a dictionary (indexed by SignalCollection) of Python slices
    # corresponding to the span of each pulsar within a Phi matrix
    def _get_slices(self, phivecs):
        ret, offset = {}, 0
        for sc, phivec in zip(self._signalcollections, phivecs):
            # assume phi is either a column vector or a square matrix
            stop = 0 if phivec is None else phivec.shape[0]
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
            return [None if phivec is None else phivec.inv(logdet)
                    for phivec in phi]
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
            slices = self._get_slices(phivecs)

            #TODO: This is messy, maybe we should clean up
            phis = [phivec for phivec in phivecs if phivec is not None]
            if np.any([phivec.ndim == 2 for phivec in phis]):
                phiinvs = [phivec.inv(logdet) for phivec in phis]
                phiinv_full = [np.diag(phi[0]) if phi[0].ndim == 1 else phi[0]
                               for phi in phiinvs]
                phiinv = sl.block_diag(*phiinv_full)
                if logdet:
                    ld = np.sum([pi[1] for pi in phiinvs])
                phidiag = np.concatenate([np.diag(phi) if phi.ndim == 2
                                          else phi for phi in phis])
            else:
                phidiag = np.concatenate(phis)
                phiinv = np.diag(1.0/phidiag)
                if logdet:
                    ld = np.sum(np.log(phidiag))

            # this will only work if all common signals are shared among all
            # the pulsars and share the same basis
            invert = None

            for csclass, csdict in self._commonsignals.items():
                for i, (cs1, csc1) in enumerate(csdict.items()):
                    for j, (cs2, csc2) in enumerate(csdict.items()):
                        if j <= i:
                            continue

                        # hoping they're all the same...
                        crossdiag = csclass.get_phicross(cs1, cs2, params)

                        if invert is None:
                            invert = np.zeros((len(crossdiag),
                                               len(csdict),
                                               len(csdict)),'d')

                        invert[:,i,j] += crossdiag
                        invert[:,j,i] += crossdiag

                    invert[:,i,i] += phidiag[slices[csc1]][csc1._idx[cs1]]

                    if logdet:
                        ld -= np.sum(np.log(
                            phidiag[slices[csc1]][csc1._idx[cs1]]))

            for k in range(len(crossdiag)):
                cf = sl.cho_factor(invert[k,:,:])
                invert[k,:,:] = sl.cho_solve(
                    cf, np.eye(invert[k,:,:].shape[0]))
                if logdet:
                    ld += np.sum(2 * np.log(np.diag(cf[0])))

            csdict = list(self._commonsignals.values())[0]
            for i, (cs1, csc1) in enumerate(csdict.items()):
                block1, idx1 = slices[csc1], csc1._idx[cs1]
                for j, (cs2, csc2) in enumerate(csdict.items()):
                    if j < i:
                        continue

                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    phiinv[block1,block2][idx1,idx2] = invert[:,i,j]
                    phiinv[block2,block1][idx2,idx1] = invert[:,i,j]

            if logdet:
                return phiinv, ld
            else:
                return phiinv
        else:
            return [None if phivec is None else phivec.inv(logdet)
                    for phivec in phivecs]

    def get_phiinv_byfreq_cliques(self, params, logdet=False, cholesky=False):
        phi = self.get_phi(params, cliques=True)

        if isinstance(phi, list):
            return [None if phivec is None else phivec.inv(logdet)
                    for phivec in phi]
        else:
            ld = 0

            # first invert all the cliques
            for clcount in range(self._clcount):
                idx = (self._cliques == clcount)

                if np.any(idx):
                    idx2 = np.ix_(idx,idx)

                    if cholesky:
                        cf = sl.cho_factor(phi[idx2])

                        if logdet:
                            ld += 2.0*np.sum(np.log(np.diag(cf[0])))

                        phi[idx2] = sl.cho_solve(
                            cf, np.identity(cf[0].shape[0]))
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

    # we use "cliques" to account for sparse non-diagonal Phi matrices
    # for each value in self._cliques, the matrix indices with that value form
    # an independent submatrix that can be inverted separately

    # reset clique index
    def _resetcliques(self, n):
        self._cliques = -1 * np.ones(n)
        self._clcount = 0

    # update clique index by considering a common signal under
    # the assumption that the corresponding "big-Phi" matrix is block diagonal
    def _setcliques(self,slices,csdict):
        # each column in idxmatrix (mind the .T) corresponds to the indices
        # that participate in a common signal for a given pulsar
        idxmatrix = np.array([csc._idx[cs] for cs, csc in csdict.items()]).T

        # each row in the updated idxmatrix corresponds to a set of "global"
        # Phi indices that are correlated across pulsars
        idxmatrix = idxmatrix + np.array([slices[csc].start
                                          for cs, csc in csdict.items()])

        # loop over vectors of common-signal-correlated global-indices
        for idxs in idxmatrix:
            # find the existing cliques assigned to these global indices
            allidx = set(self._cliques[idxs])
            maxidx = max(allidx)

            if maxidx == -1:
                # if no clique is found, create a new one, and assign it
                # to the indices in idx

                self._cliques[idxs] = self._clcount

                # I don't think this code is ever exercised...
                # if maxidx == -1, then allidx = [-1]
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques,allidx)] = \
                        self._clcount

                self._clcount = self._clcount + 1
            else:
                # if we find at least one clique, assign all indices in idx
                # to the maximum clique index

                self._cliques[idxs] = maxidx

                # since cliques are "contagious", reassign all the other
                # clique indices that we found to maxidx
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques,allidx)] = maxidx

    # add cliques from individual pulsar phis; these will never overlap
    # TO DO: at this point Phi could be defined as a smarter KernelMatrix!
    def _setpulsarcliques(self, slices, phis):
        for sc, phi in zip(self._signalcollections, phis):
            if phi is not None:
                for clindex in range(getattr(phi, '_clcount', 0)):
                    phiind = np.where(phi._cliques == clindex)[0]

                    if len(phiind) > 0:
                        try:
                            self._cliques[slices[sc].start+phiind] = (
                                self._clcount)
                            self._clcount = self._clcount + 1
                        except:
                            print(self._cliques.shape)
                            print("phiind",phiind,len(phiind))
                            print(slices)
                            raise

    def get_phi(self, params, cliques=False):
        phis = [signalcollection.get_phi(params) for
                signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            if np.any([phi.ndim == 2 for phi in phis if phi is not None]):
                # if we have any dense matrices,
                Phi = sl.block_diag(*[np.diag(phi) if phi.ndim == 1 else phi
                                      for phi in phis
                                      if phi is not None])
            else:
                Phi = np.diag(np.concatenate([phi for phi in phis
                                              if phi is not None]))

            # get a dictionary of slices locating each pulsar in Phi matrix
            slices = self._get_slices(phis)

            # self._cliques is a vector of the same size as the Phi matrix
            # for each Phi index i, self._cliques[i] is -1 if row/column
            # belong to no clique, or it gives the clique number otherwise
            if cliques:
                self._resetcliques(Phi.shape[0])
                self._setpulsarcliques(slices, phis)

            # iterate over all common signal classes
            for csclass, csdict in self._commonsignals.items():
                # first figure out which indices are used in this common signal
                # and update the clique index
                if cliques:
                    self._setcliques(slices, csdict)

                # now iterate over all pairs of common signal instances
                pairs = itertools.combinations(csdict.items(),2)

                for (cs1, csc1), (cs2, csc2) in pairs:
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    Phi[block1,block2][idx1,idx2] += crossdiag
                    Phi[block2,block1][idx2,idx1] += crossdiag

            return Phi
        else:
            return phis

    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct:ct+n] if n > 1 else float(xs[ct])
            ct += n
        return ret

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

            self._set_cache_parameters()

        def __add__(self, other):
            return PTA([self, other])

        #TODO: this could be implemented more cleanly
        def _set_cache_parameters(self):
            """ Sets the cache for various signal types."""

            self.white_params = []
            self.basis_params = []
            self.delay_params = []
            for signal in self._signals:
                if signal.signal_type == 'white noise':
                    self.white_params.extend(signal.ndiag_params)
                elif signal.signal_type in ['basis', 'common basis']:
                    self.basis_params.extend(signal.basis_params)
                elif signal.signal_type == 'deterministic':
                    self.delay_params.extend(signal.delay_params)
                else:
                    msg = '{} signal type not recognized! Caching '.format(
                        signal.signal_type)
                    msg += 'may not work correctly for this signal.'
                    logger.error(msg)

        # a candidate for memoization
        @property
        def params(self):
            return sorted({param for signal in self._signals for param
                           in signal.params}, key=lambda par: par.name)

        @property
        def param_names(self):
            ret = []
            for p in self.params:
                if p.size:
                    for ii in range(0, p.size):
                        ret.append(p.name+'_{}'.format(ii))
                else:
                    ret.append(p.name)
            return ret

        def set_default_params(self, params):
            for signal in self._signals:
                signal.set_default_params(params)

        def _combine_basis_columns(self, signals):
            """Given a set of Signal objects, each of which may return an
            Fmat (through get_basis()), return a dict (indexed by signal)
            of integer arrays that map individual Fmat columns to the
            combined Fmat.

            Note: The Fmat returned here is simply meant to initialize the
            matrix to save computations when calling `get_basis` later.
            """

            idx, Fmatlist, hashlist = {}, [], []
            cc = 0
            for signal in signals:
                Fmat = signal.get_basis()

                if Fmat is not None and not signal.basis_params:
                    idx[signal] = []

                    for i, column in enumerate(Fmat.T):
                        colhash = hash(column.tostring())
                        try:
                            j = hashlist.index(colhash)
                            idx[signal].append(j)
                        except ValueError:
                            idx[signal].append(cc)
                            Fmatlist.append(column)
                            hashlist.append(colhash)
                            cc += 1
                elif Fmat is not None and signal.basis_params:
                    nf = Fmat.shape[1]
                    idx[signal] = list(np.arange(cc, cc+nf))
                    cc += nf

            ncol = len(np.unique(sum(idx.values(), [])))
            nrow = len(Fmatlist[0])
            return ({key: np.array(idx[key]) for key in idx.keys()},
                    np.zeros((nrow, ncol)))

        # goofy way to cache _idx
        def __getattr__(self, par):
            if par in ('_idx', '_Fmat'):
                self._idx, self._Fmat = self._combine_basis_columns(
                    self._signals)
                return getattr(self,par)
            else:
                raise AttributeError("{} object has no attribute {}".format(
                    self.__class__,par))

        @cache_call('white_params')
        def get_ndiag(self, params):
            ndiags = [signal.get_ndiag(params) for signal in self._signals]
            return sum(ndiag for ndiag in ndiags if ndiag is not None)

        @cache_call('delay_params')
        def get_delay(self, params):
            delays = [signal.get_delay(params) for signal in self._signals]
            return sum(delay for delay in delays if delay is not None)

        @cache_call('delay_params')
        def get_detres(self, params):
            return self._residuals - self.get_delay(params)

        @cache_call('basis_params')
        def get_basis(self, params={}):
            for signal in self._signals:
                if signal in self._idx:
                    self._Fmat[:, self._idx[signal]] = signal.get_basis(params)
            return self._Fmat

        def get_phiinv(self, params):
            return self.get_phi(params).inv()

        def get_phi(self, params):
            phi = KernelMatrix(self._Fmat.shape[1])
            for signal in self._signals:
                if signal in self._idx:
                    phi = phi.add(signal.get_phi(params), self._idx[signal])
            return phi

        @cache_call(['basis_params', 'white_params', 'delay_params'])
        def get_TNr(self, params):
            Nvec = self.get_ndiag(params)
            T = self.get_basis(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=T)

        @cache_call(['basis_params', 'white_params'])
        def get_TNT(self, params):
            Nvec = self.get_ndiag(params)
            T = self.get_basis(params)
            return Nvec.solve(T, left_array=T)

        @cache_call(['white_params', 'delay_params'])
        def get_rNr_logdet(self, params):
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=res, logdet=True)

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

        def add_kwarg(self, **kwargs):
            self._defaults.update(kwargs)

        @property
        def params(self):
            # if we extract the ConstantParameter value above, we would not
            # need a special case here
            return [par for par in self._params.values() if not
                    isinstance(par, ConstantParameter)]

    return Function


def get_funcargs(func):
    """Convienience function to get args and kwargs of any function."""
    argspec = inspect.getargspec(func)
    if argspec.defaults is None:
        args = argspec.args
        kwargs = []
    else:
        args = argspec.args[:(len(argspec.args)-len(argspec.defaults))]
        kwargs = argspec.args[-len(argspec.defaults):]

    return args, kwargs


def function(func):
    """Decorator for Function."""

    funcargs, _ = get_funcargs(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fargs = {funcargs[ct]: val for ct, val in
                 enumerate(args[:len(funcargs)])}
        fargs.update(kwargs)
        if not np.all([fa in fargs.keys() for fa in funcargs]):
            return Function(func, **kwargs)
        for kw, arg in kwargs.items():
            if ((isinstance(arg, type) and issubclass(
                arg, (Parameter, ConstantParameter))) or isinstance(
                    arg, (Parameter, ConstantParameter))):
                return Function(func, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def cache_call(attrs, limit=2):
    """Cache function that allows for subsets of parameters to be keyed."""

    # convert to list of lists if only one attribute used
    if not isinstance(attrs, list):
        attrs = [attrs]

    def cache_decorator(func):

        def wrapper(self, params={}):

            # get the relevant parameters to be cached
            keys = sum([getattr(self, attr) for attr in attrs], [])
            ret = []
            # TODO: this deals with vector parameters but could be cleaner...
            for key in keys:
                if key in params:
                    if np.ndim(params[key]) > 0:
                        ret.append((key, tuple(params[key])))
                    else:
                        ret.append((key, params[key]))
            key = tuple(ret)
            #key = tuple([(key, params[key]) for key in keys if key in params])

            # make sure the cache is part of the object
            if not hasattr(self, '_cache_'+func.__name__):
                msg = 'Create cache {} for signal {}'.format(
                    func.__name__, self.__class__)
                logger.debug(msg)
                setattr(self, '_cache_'+func.__name__, {})
                setattr(self, '_cache_list_'+func.__name__, [])
            cache = getattr(self, '_cache_'+func.__name__)
            cache_list = getattr(self, '_cache_list_'+func.__name__)

            if key not in cache:
                msg = 'Setting cache for {} in {}: {}'.format(
                    attrs, self.__class__, key)
                logger.debug(msg)
                cache_list.append(key)
                cache[key] = func(self, params)
                if len(cache_list) > limit:
                    del cache[cache_list.pop(0)]
            return cache[key]
        return wrapper

    return cache_decorator


class KernelMatrix(np.ndarray):
    def __new__(cls, init):
        if isinstance(init, int):
            ret = np.zeros(init, 'd').view(cls)
        else:
            ret = init.view(cls)

        if ret.ndim == 2:
            ret._cliques = -1 * np.ones(ret.shape[0])
            ret._clcount = 0

        return ret

    # see PTA._setcliques
    def _setcliques(self, idxs):
        allidx = set(self._cliques[idxs])
        maxidx = max(allidx)

        if maxidx == -1:
            self._cliques[idxs] = self._clcount
            self._clcount = self._clcount + 1
        else:
            self._cliques[idxs] = maxidx
            if len(allidx) > 1:
                self._cliques[np.in1d(self._cliques,allidx)] = maxidx

    def add(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] += other
        else:
            if other.ndim == 1:
                self[idx, idx] += other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] += other

        return self

    def set(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] = other
        else:
            if other.ndim == 1:
                self[idx, idx] = other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] = other

        return self

    def inv(self, logdet=False):
        if self.ndim == 1:
            inv = 1.0/self

            if logdet:
                return inv, np.sum(np.log(self))
            else:
                return inv
        else:
            try:
                cf = sl.cho_factor(self)
                inv = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                if logdet:
                    ld = 2.0*np.sum(np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                u, s, v = np.linalg.svd(self)
                inv = np.dot(u/s, u.T)
                if logdet:
                    ld = np.sum(np.log(s))
            if logdet:
                return inv, ld
            else:
                return inv


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
        except:
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

        if np.any(nvec != 0):
            s1 = set(np.arange(len(nvec)))
            s2 = set(np.concatenate([np.arange(len(nvec))[slc]
                                     for slc in slices]))
            sd = s1.difference(s2)
            self._idx = np.array([s for s in sd])

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
        if len(self._idx) > 0:
            ZNXr = np.dot(Z[self._idx,:].T, X[self._idx,:] /
                          self._nvec[self._idx, None])
        else:
            ZNXr = 0
        for slc, block in zip(self._slices, self._blocks):
            Zblock = Z[slc, :]
            Xblock = X[slc, :]

            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block+np.diag(self._nvec[slc]))
                bx = sl.cho_solve(cf, Xblock)
            else:
                bx = Xblock / self._nvec[slc][:, None]
            ZNX += np.dot(Zblock.T, bx)
        ZNX += ZNXr
        return ZNX.squeeze() if len(ZNX) > 1 else float(ZNX)

    def _solve_NX(self, X):
        """Solves :math:`N^{-1}X`, where :math:`X`
        is a 1-d or 2-d array.
        """
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        NX = X / self._nvec[:,None]
        for slc, block in zip(self._slices, self._blocks):
            Xblock = X[slc, :]
            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block+np.diag(self._nvec[slc]))
                NX[slc] = sl.cho_solve(cf, Xblock)
        return NX.squeeze()

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        if len(self._idx) > 0:
            logdet = np.sum(np.log(self._nvec[self._idx]))
        else:
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
