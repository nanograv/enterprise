# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""

import time
import timeit
import platform

try:
    import cpuinfo

    def cpu_model():
        return cpuinfo.get_cpu_info()["brand_raw"]


except ModuleNotFoundError:

    def cpu_model():
        return "unknown CPU (for better info install cpuinfo)"


import collections

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

import itertools
import logging

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import six
from sksparse.cholmod import cholesky, CholmodError

# these are defined in parameter.py, but currently imported
# in various places from signal_base.py
from enterprise.signals.parameter import Function  # noqa: F401
from enterprise.signals.parameter import function  # noqa: F401
from enterprise.signals.parameter import ConstantParameter
from enterprise.signals.utils import KernelMatrix

from enterprise import __version__
from sys import version

_py_version = version.split(" ")[0]

# logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaSignal(type):
    """Metaclass for Signals. Allows addition of ``Signal`` classes."""

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection([self, other])
        elif isinstance(other, MetaCollection):
            return SignalCollection([self] + other._metasignals)
        else:
            raise TypeError


class MetaCollection(type):
    """Metaclass for Signal collections. Allows addition of
    ``SignalCollection`` classes.
    """

    def __add__(self, other):
        if isinstance(other, MetaSignal):
            return SignalCollection(self._metasignals + [other])
        elif isinstance(other, MetaCollection):
            return SignalCollection(self._metasignals + other._metasignals)
        else:
            raise TypeError


@six.add_metaclass(MetaSignal)
class Signal(object):
    """Base class for Signal objects."""

    is_timing_model = False  # See TimingModel

    def __init__(self, psr):
        self.psrname = psr.name

    @property
    def params(self):
        # return only nonconstant parameters
        return [par for par in self._params.values() if not isinstance(par, ConstantParameter)]

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
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
                msg = "Setting {} to {}".format(par.name, params[par.name])
                logger.info(msg)
                self._params[kw].value = params[par.name]
            elif par.name not in params and isinstance(par, ConstantParameter):
                if par.value is None:
                    msg = "{} not set! Check parameter dict.".format(par.name)
                    logger.warning(msg)

    def get_ndiag(self, params):
        """Returns the diagonal of the white noise vector `N`.

        This method also supports block diagonal sparse matrices.
        """
        return None

    def get_delay(self, params):
        """Returns the waveform of a deterministic signal."""
        return 0

    def get_basis(self, params=None):
        """Returns the basis array of shape N_toa x N_basis."""
        return None

    def get_phi(self, params):
        """Returns a diagonal covariance matrix of the basis amplitudes."""
        return None

    def get_phiinv(self, params):
        """Returns inverse of the covaraince of basis amplitudes."""
        return None

    def get_logsignalprior(self, params):
        """Returns an additional prior/likelihood terms associated with a signal."""
        return 0


class CommonSignal(Signal):
    """Base class for CommonSignal objects."""

    def get_phiinv(self, params):
        msg = "You probably shouldn't be calling get_phiinv() "
        msg += "on a common red-noise signal."
        raise RuntimeError(msg)

    @classmethod
    def get_phicross(cls, signal1, signal2, params):
        return None


"""Procedures to calculate likelihoods:

Notation:
r = vector of TOA residuals
M = design matrix.  Row = TOA number, column = design parameter
F = Fourier (or other) basis.  Row = TOA number, column = parameter
T = (F | M)
E = diagonal matrix of infinities giving lack of knowledge about model parameters
chi = covariance matrix of parameters.  Xavi's notes call this phi.
phi = block diagonal matrix of chi and E.  Xavi's notes call this B.
N = white noise matrix, generally in Sherman-Morrison form
C = N + T phi T^T
C^-1 = N^-1 - N^-1 T (phi^-1 + T^T N^-1 T)^-1 T^T N^-1
The log of the likelihood is -(1/2) ln det(2 pi N) + (1/2) r^T C^-1 r
By the matrix determinant lemma, det(C) = det(phi^-1 + T^T N^-1 T) det(phi) det(N)

The one step procedure:

Compute and cache:

T N^-1 r (This is called TNr, and the same convention applies throughout)
T^T N^-1 T
r N^-1 r
det(N)

For each set of parameters, compute:

phi^-1
det(phi)
Sigma = phi^-1 + TNT
Sigma^-1 TNr and det(Sigma) by Cholesky decomposition
(TNr)^T Sigma^-1 TNr
Now r^T C^-1 r = rNr - (TNr)^T Sigma^-1 TNr
and det(C) = det(Sigma) det(phi) det(N).
lnlikelihood = -(1/2)(rCr + ln det(C)).  We do not include the factor (2pi)^N_TOA

For the two step procedure, we define also:

D = N + M E M^T
D^-1 = N^-1 - N^-1 M (M^T N^-1 M)^-1 M^T N^-1
C = D + F chi F^T
C^-1 = D^-1 - D^-1 F (chi^-1 + F^T D^-1 F)^-1 F^T D^-1

We first compute everything that doesn't depend on the parameters:
M^T N^-1 M
M N^-1 r
r N^-1 r
(MNM)^-1 and det(MNM) by Cholesky decomposition
F^T N^-1 F
M^T N^-1 F
(MNM)^-1 MNF
F^T D^-1 F = FNF - (MNF)^T (MNM)^-1 MNF
r^T D^-1 r = r N^-1 r  - (MNr)^T (MNM)^-1 MNr
F^T D^-1 r = F^T N^-1 r - (MNMMNF)^T MNr
det(D) = det(MNM) det(N).  We don't include the infinite det(E) here.

Then for each set of parameters we compute:
chi^-1
det(chi)
Sigma = chi^-1 + FDF
Sigma^-1 and det(Sigma) by Cholesky decomposition.
Now r C^-1 r = rDr - (FDr)^T Sigma^-1 FDr
and det(C) = det(Sigma) det(chi) det(D)

If there are no timing parameters, M and everything that depends on it will be None, and then D = N.
If there are no model parameters, F and everything that depends on it will be None, and then D = C.

"""


class LogLikelihood(object):
    def __init__(self, pta, cholesky_sparse=False, timer=time.process_time):
        self.pta = pta
        self._cholesky_sparse = cholesky_sparse
        self.timer = timer
        self._cached_FDFs = None
        self._cached_FDrs = None
        self.cholesky_time = 0
        self.cholesky_calls = 0
        self._cached_factor = None

    # Cache the FDF matrix and FDr vector.  This happens only if there are correlations.
    def get_FDF(self, params):
        FDFs = self.pta.get_FDF(params)
        # If this is the first time or anything has changed since last time, compute it
        # pta.get_FDF returns one object per pulsar, so the lists will be the same length
        if not self._cached_FDFs or any(x is not y for x, y in zip(FDFs, self._cached_FDFs)):
            self._cached_FDFs = FDFs
            self.FDF = sps.block_diag(FDFs, "csc") if self._cholesky_sparse else sl.block_diag(*FDFs)
        return self.FDF

    def get_FDr(self, params):
        FDrs = self.pta.get_FDr(params)
        # See get_FDF
        if not self._cached_FDrs or any(x is not y for x, y in zip(FDrs, self._cached_FDrs)):
            self._cached_FDrs = FDrs
            self.FDr = np.concatenate(FDrs)
        return self.FDr

    # We don't cache rDr because it would take just as long to check whether it was needed
    # updating as to update it

    # Sigma = chi^-1 + FDF.
    def _make_sigma(self, params, chiinv):
        return self.get_FDF(params) + (sps.csc_matrix(chiinv) if self._cholesky_sparse else chiinv)

    def __call__(self, xs, phiinv_method="cliques"):
        # Method for inverting chi called phiinv_method for compatibility
        # map parameter vector if needed
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        # get -0.5 * (rDr + logdet_D) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        lnlike = -0.5 * np.sum([ell for ell in self.pta.get_rDr_logdet(params)])

        # get extra prior/likelihoods
        lnlike += sum(self.pta.get_logsignalprior(params))

        chiinvs = self.pta.get_chiinv(params, logdet=True, method=phiinv_method)

        if self.pta._commonsignals:
            # This happens for correlations.
            chiinv, logdet_chi = chiinvs
            Sigma = self._make_sigma(params, chiinv)
            FDr = self.get_FDr(params)

            start = self.timer()
            if self._cholesky_sparse:
                try:
                    # If we have a cached Cholesky factor, reuse it without finding
                    # the best perturbation anew.  This assumes that any changes don't affect
                    # the pattern of nonzero elements in Sigma.
                    if self._cached_factor:
                        self._cached_factor.cholesky_inplace(Sigma)
                    else:
                        self._cached_factor = cholesky(Sigma, ordering_method="best")  # First time
                    cf = self._cached_factor
                except CholmodError:
                    return -np.inf
            else:  # Not using sparse methods
                try:
                    cf = sl.cho_factor(Sigma)  # returns tuple with flag saying lower triangular
                except np.linalg.LinAlgError:
                    return -np.inf

            this_time = self.timer() - start
            self.cholesky_time += this_time
            self.cholesky_calls += 1

            if self._cholesky_sparse:
                expval = cf(FDr)
                logdet_sigma = cf.logdet()
            else:
                expval = sl.cho_solve(cf, FDr)
                logdet_sigma = 2 * np.sum(np.log(np.diag(cf[0])))  # A = L L^T so det = det(L)^2

            lnlike += 0.5 * (np.dot(FDr, expval) - logdet_sigma - logdet_chi)

        else:
            # This happens for anything that doesn't include correlations
            # Then chiinvs is a list of matrix (or its diagonal), logdet for each pulsar
            # and we compute each pulsar and total them.
            for FDr, FDF, pl in zip(self.pta.get_FDr(params), self.pta.get_FDF(params), chiinvs):
                if FDr is None:
                    continue

                chiinv, logdet_chi = pl
                Sigma = FDF + (np.diag(chiinv) if chiinv.ndim == 1 else chiinv)

                start = self.timer()

                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, FDr)
                except np.linalg.LinAlgError:
                    return -np.inf

                this_time = self.timer() - start
                self.cholesky_time += this_time
                self.cholesky_calls += 1

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
                lnlike += 0.5 * (np.dot(FDr, expval) - logdet_sigma - logdet_chi)

        return lnlike


# This is just LogLikelihood with sparse_cholesky=True as the default
class LogLikelihoodSparse(LogLikelihood):
    def __init__(self, pta, cholesky_sparse=True, **kwargs):
        LogLikelihood.__init__(self, pta, cholesky_sparse=cholesky_sparse, **kwargs)


class OldLogLikelihood(object):
    def __init__(self, pta, timer=time.process_time):
        self.pta = pta
        self.timer = timer
        self.uses_1e40 = True  # Say that we include "infinite" terms
        self.cholesky_time = 0
        self.cholesky_calls = 0

    def _make_sigma(self, TNTs, phiinv):
        return sps.block_diag(TNTs, "csc") + sps.csc_matrix(phiinv)

    def __call__(self, xs, phiinv_method="cliques"):
        # map parameter vector if needed
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        loglike = 0

        # phiinvs will be a list or may be a big matrix if spatially
        # correlated signals
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=True, method=phiinv_method)

        # get -0.5 * (rNr + logdet_N) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))

        # red noise piece
        if self.pta._commonsignals:
            phiinv, logdet_phi = phiinvs

            Sigma = self._make_sigma(TNTs, phiinv)
            TNr = np.concatenate(TNrs)

            start = self.timer()
            try:
                cf = cholesky(Sigma)
            except CholmodError:
                return -np.inf

            this_time = self.timer() - start
            self.cholesky_time += this_time
            self.cholesky_calls += 1

            expval = cf(TNr)

            logdet_sigma = cf.logdet()

            loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)
        else:
            for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
                if TNr is None:
                    continue

                phiinv, logdet_phi = pl
                Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                start = self.timer()
                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                except:
                    return -np.inf

                this_time = self.timer() - start
                self.cholesky_time += this_time
                self.cholesky_calls += 1

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))

                loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike


class LikelihoodsDifferentError(Exception):
    def __init__(self, object1, likelihood1, object2, likelihood2):
        Exception.__init__(
            self,
            "Likelihoods too different, {} gave {} while {} gave {}".format(object1, likelihood1, object2, likelihood2),
        )


# Compare different ways of computing the likelihood for time and consistency
# We make LogLikelihood objects using then given classes or constructors
# and call all of them.  The return value is from the first object.
# We call the different likelihood objects in all possible orders, going to
# the next permutation each  time that we are called.  This prevents later objects
# from looking better because earlier objects have cached useful values.
# However, this could be defeated by a resonance between this process
# and the process calling us.  If you have n objects, you should do a multiple of
# n! tests if you want to be precise.
class CompareLogLikelihood(object):
    def __init__(
        self,
        pta,
        classes=(LogLikelihood, OldLogLikelihood),
        timer=time.process_time,
        tolerance=1e-2,  # Absolute tolerance for likelihood differences
    ):
        self.pta = pta
        self.constructors = classes
        self.n_objects = len(classes)
        self.likelihoods = np.zeros(self.n_objects)  # make list for later
        self.timer = timer
        self.tolerance = tolerance
        self.objects = [x(pta, timer=timer) for x in classes]
        self.reset()

    # Reset or initialize differences in timers
    def reset(self):
        self.max_differences = np.zeros((self.n_objects, self.n_objects))  # Differences between pairs of results
        self.times = np.zeros(self.n_objects)  # Time in each object
        self.counts = np.zeros(self.n_objects, dtype=int)  # Count of calls to each object
        self.orders = tuple(itertools.permutations(list(range(self.n_objects))))  # Possible orders
        self.order_pointer = 0  # Order to use next
        for object in self.objects:  # Reset timers in sub-objects
            object.cholesky_calls = object.cholesky_time = 0

    def __call__(self, xs, **kwargs):
        # on jth call, go in order j,j+1...n-1, 0, 1, .. j-1.  See above
        order = self.orders[self.order_pointer]
        self.order_pointer = (self.order_pointer + 1) % len(self.orders)  # Ready for next
        for i in order:
            self.times[i] += timeit.timeit(lambda: self.call_object(i, xs, **kwargs), timer=self.timer, number=1)
            self.counts[i] += 1
        for i in range(self.n_objects):
            for j in range(i):
                diff = abs(self.likelihoods[i] - self.likelihoods[j])
                self.max_differences[i, j] = max(self.max_differences[i, j], diff)
                if diff > self.tolerance:
                    raise LikelihoodsDifferentError(
                        self.objects[i], self.likelihoods[i], self.objects[j], self.likelihoods[j]
                    )
        return self.likelihoods[0]

    # Call one of our objects and store the resulting likelihood
    def call_object(self, i, xs, **kwargs):
        likelihood = self.objects[i](xs, **kwargs)
        # If object used 1e40 as infinity and gave determinants using that,
        # remove them now.
        if getattr(self.objects[i], "uses_1e40", False):
            for collection in self.pta.pulsarmodels:
                M = collection.get_basis_M()  # Matrix with with timing model rows
                if M is not None:
                    likelihood += 0.5 * M.shape[1] * np.log(1e40)
                else:
                    continue
        self.likelihoods[i] = likelihood

    def report(self):
        print("The maximum difference between any two loglikelihoods was {:.3g}".format(np.amax(self.max_differences)))
        print(
            "Running on",
            platform.node(),
            cpu_model(),
            "with",
            len(self.pta.pulsars),
            "pulsars",
            "using timer",
            self.timer.__name__,
        )
        for i in range(self.n_objects):
            print(self.constructors[i].__name__, end=" ")
            if self.counts[i] > 0:
                print(
                    "{} calls, {:.2f} ms/call,".format(int(self.counts[i]), 1000 * self.times[i] / self.counts[i]),
                    end=" ",
                )
            else:
                print("not called,", end=" ")
            if self.objects[i].cholesky_calls > 0:
                print(
                    "{:.2f} of this in a total of {} cholesky calls".format(
                        1000 * self.objects[i].cholesky_time / self.counts[i], self.objects[i].cholesky_calls
                    )
                )
            else:
                print("no cholesky calls")


""" Example usage of CompareLogLikelihood class:
c = signal_base.CompareLogLikelihood(pta)
c(x0)
c.reset()                       # Reset after first use to not include setup time
c(x0)                           # Or sample as much as you want
c.report()
"""


class PTA(object):
    # lnlikelihood is generally a class, but can be anything that can
    # be called with the pta object to return a likelihood object
    def __init__(self, init, lnlikelihood=LogLikelihood):
        if isinstance(init, Sequence):
            self._signalcollections = list(init)
        else:
            self._signalcollections = [init]

        self.lnlikelihood = lnlikelihood

        # set signal dictionary
        self._set_signal_dict()

    def __add__(self, other):
        if hasattr(other, "_signalcollections"):
            return PTA(self._signalcollections + other._signalcollections, lnlikelihood=self.lnlikelihood)
        else:
            return PTA(self._signalcollections + [other], lnlikelihood=self.lnlikelihood)

    @property
    def params(self):
        ret = set()

        for signalcollection in self._signalcollections:
            for param in signalcollection.params:
                for par in param.params:
                    ret.add(par)

        return sorted(list(ret), key=lambda par: par.name)

        # return sorted({par for signalcollection in self._signalcollections
        #                    for par in signalcollection.params},
        #               key=lambda par: par.name)

    @property
    def param_names(self):
        ret = []
        for p in self.params:
            if p.size:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret

    @property
    def pulsarmodels(self):
        return self._signalcollections

    def get_TNr(self, params):
        return [signalcollection.get_TNr(params) for signalcollection in self._signalcollections]

    def get_TNT(self, params):
        return [signalcollection.get_TNT(params) for signalcollection in self._signalcollections]

    def get_rNr_logdet(self, params):
        return [signalcollection.get_rNr_logdet(params) for signalcollection in self._signalcollections]

    # new pieces:
    def get_MNr(self, params):
        return [signalcollection.get_MNr(params) for signalcollection in self._signalcollections]

    def get_FNr(self, params):
        return [signalcollection.get_FNr(params) for signalcollection in self._signalcollections]

    def get_MNM(self, params):
        return [signalcollection.get_MNM(params) for signalcollection in self._signalcollections]

    def get_MNM_cholesky(self, params):
        return [signalcollection.get_MNM_cholesky(params) for signalcollection in self._signalcollections]

    def get_MNM_logdet(self, params):
        return [signalcollection.get_MNM_logdet(params) for signalcollection in self._signalcollections]

    def get_FNF(self, params):
        return [signalcollection.get_FNF(params) for signalcollection in self._signalcollections]

    def get_MNF(self, params):
        return [signalcollection.get_MNF(params) for signalcollection in self._signalcollections]

    def get_MNMMNF(self, params):
        return [signalcollection.get_MNMMNF(params) for signalcollection in self._signalcollections]

    def get_FDF(self, params):
        return [signalcollection.get_FDF(params) for signalcollection in self._signalcollections]

    def get_FDr(self, params):
        return [signalcollection.get_FDr(params) for signalcollection in self._signalcollections]

    def get_rDr_logdet(self, params):
        return [signalcollection.get_rDr_logdet(params) for signalcollection in self._signalcollections]

    # back to other pieces here:
    def get_residuals(self):
        return [signalcollection._residuals for signalcollection in self._signalcollections]

    def get_ndiag(self, params={}):
        return [signalcollection.get_ndiag(params) for signalcollection in self._signalcollections]

    def get_delay(self, params={}):
        return [signalcollection.get_delay(params) for signalcollection in self._signalcollections]

    def get_logsignalprior(self, params):
        return [signalcollection.get_logsignalprior(params) for signalcollection in self._signalcollections]

    def set_default_params(self, params):
        for sc in self._signalcollections:
            sc.set_default_params(params)

    def get_basis(self, params={}):
        return [signalcollection.get_basis(params) for signalcollection in self._signalcollections]

    @property
    def _lnlikelihood(self):
        # instantiate on first use
        if not hasattr(self, "_lnlike"):
            self._lnlike = self.lnlikelihood(self)

        return self._lnlike

    def get_lnlikelihood(self, params, **kwargs):
        return self._lnlikelihood(params, **kwargs)

    @property
    def _commonsignals(self):
        # cache the computation if we don't have it yet
        if not hasattr(self, "_cs"):
            commonsignals = collections.defaultdict(collections.OrderedDict)

            for signalcollection in self._signalcollections:
                # TODO: need a better signal that a
                # signalcollection provides a basis

                if signalcollection._Fmat is not None:
                    for signal in signalcollection._signals:
                        # if the CommonSignal is coefficient based we don't
                        # need to worry about it for get_phi and get_phiinv
                        if isinstance(signal, CommonSignal) and not getattr(signal, "_coefficients", {}):
                            commonsignals[signal.__class__][signal] = signalcollection

            # drop common signals that appear only once
            self._cs = {csclass: csdict for csclass, csdict in commonsignals.items() if len(csdict) > 1}

        return self._cs

    # return a dictionary (indexed by SignalCollection) of Python slices
    # corresponding to the span of each pulsar within a Phi matrix
    def _get_slices(self, phivecs):
        ret, offset = {}, 0
        for sc, phivec in zip(self._signalcollections, phivecs):
            # assume phi is either a column vector or a square matrix
            stop = 0 if phivec is None else phivec.shape[0]
            ret[sc] = slice(offset, offset + stop)
            offset = ret[sc].stop

        return ret

    def get_phiinv(self, params, logdet=False, method="cliques"):
        if method == "cliques":
            return self.get_phiinv_byfreq_cliques(params, logdet)
        elif method == "partition":
            return self.get_phiinv_byfreq_partition(params, logdet)
        elif method == "sparse":
            return self.get_phiinv_sparse(params, logdet)
        else:
            raise NotImplementedError

    def get_phiinv_sparse(self, params, logdet=False):
        phi = self.get_phi(params)

        if isinstance(phi, list):
            return [None if phivec is None else phivec.inv(logdet) for phivec in phi]
        else:
            phisparse = sps.csc_matrix(phi)
            cf = cholesky(phisparse)

            if logdet:
                return (cf.inv(), cf.logdet())
            else:
                return cf.inv()

    def get_phiinv_byfreq_partition(self, params, logdet=False):
        phivecs = [signalcollection.get_phi(params) for signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            slices = self._get_slices(phivecs)

            # TODO: This is messy, maybe we should clean up
            phis = [phivec for phivec in phivecs if phivec is not None]
            if np.any([phivec.ndim == 2 for phivec in phis]):
                phiinvs = [phivec.inv(logdet) for phivec in phis]
                phiinv_full = [np.diag(phi[0]) if phi[0].ndim == 1 else phi[0] for phi in phiinvs]
                phiinv = sl.block_diag(*phiinv_full)
                if logdet:
                    ld = np.sum([pi[1] for pi in phiinvs])
                phidiag = np.concatenate([np.diag(phi) if phi.ndim == 2 else phi for phi in phis])
            else:
                phidiag = np.concatenate(phis)
                phiinv = np.diag(1.0 / phidiag)
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
                            invert = np.zeros((len(crossdiag), len(csdict), len(csdict)), "d")

                        if crossdiag.ndim == 2:
                            raise NotImplementedError(
                                "get_phiinv with method='partition' does not " "support dense phi matrices."
                            )

                        invert[:, i, j] += crossdiag
                        invert[:, j, i] += crossdiag

                    invert[:, i, i] += phidiag[slices[csc1]][csc1._idx[cs1]]

                    if logdet:
                        ld -= np.sum(np.log(phidiag[slices[csc1]][csc1._idx[cs1]]))

            for k in range(len(crossdiag)):
                cf = sl.cho_factor(invert[k, :, :])
                invert[k, :, :] = sl.cho_solve(cf, np.eye(invert[k, :, :].shape[0]))
                if logdet:
                    ld += np.sum(2 * np.log(np.diag(cf[0])))

            csdict = list(self._commonsignals.values())[0]
            for i, (cs1, csc1) in enumerate(csdict.items()):
                block1, idx1 = slices[csc1], csc1._idx[cs1]
                for j, (cs2, csc2) in enumerate(csdict.items()):
                    if j < i:
                        continue

                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    phiinv[block1, block2][idx1, idx2] = invert[:, i, j]
                    phiinv[block2, block1][idx2, idx1] = invert[:, i, j]

            if logdet:
                return phiinv, ld
            else:
                return phiinv
        else:
            return [None if phivec is None else phivec.inv(logdet) for phivec in phivecs]

    def get_chiinv(self, params, logdet=False, method="cliques"):
        if method == "cliques":
            return self.get_chiinv_byfreq_cliques(params, logdet)
        elif method == "partition":
            return self.get_chiinv_byfreq_partition(params, logdet)
        elif method == "sparse":
            return self.get_chiinv_sparse(params, logdet)
        else:
            raise NotImplementedError

    def get_chiinv_byfreq_partition(self, params, logdet=False):
        chivecs = [signalcollection.get_chi(params) for signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            slices = self._get_slices(chivecs)

            # TODO: This is messy, maybe we should clean up
            chis = [chivec for chivec in chivecs if chivec is not None]
            if np.any([chivec.ndim == 2 for chivec in chis]):
                chiinvs = [chivec.inv(logdet) for chivec in chis]
                chiinv_full = [np.diag(chi[0]) if chi[0].ndim == 1 else chi[0] for chi in chiinvs]
                chiinv = sl.block_diag(*chiinv_full)
                if logdet:
                    ld = np.sum([pi[1] for pi in chiinvs])
                chidiag = np.concatenate([np.diag(chi) if chi.ndim == 2 else chi for chi in chis])
            else:
                chidiag = np.concatenate(chis)
                chiinv = np.diag(1.0 / chidiag)
                if logdet:
                    ld = np.sum(np.log(chidiag))

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
                            invert = np.zeros((len(crossdiag), len(csdict), len(csdict)), "d")

                        if crossdiag.ndim == 2:
                            raise NotImplementedError(
                                "get_phiinv with method='partition' does not " "support dense phi matrices."
                            )

                        invert[:, i, j] += crossdiag
                        invert[:, j, i] += crossdiag

                    invert[:, i, i] += chidiag[slices[csc1]][csc1._idx[cs1]]

                    if logdet:
                        ld -= np.sum(np.log(chidiag[slices[csc1]][csc1._idx[cs1]]))

            for k in range(len(crossdiag)):
                cf = sl.cho_factor(invert[k, :, :])
                invert[k, :, :] = sl.cho_solve(cf, np.eye(invert[k, :, :].shape[0]))
                if logdet:
                    ld += np.sum(2 * np.log(np.diag(cf[0])))

            csdict = list(self._commonsignals.values())[0]
            for i, (cs1, csc1) in enumerate(csdict.items()):
                block1, idx1 = slices[csc1], csc1._idx[cs1]
                for j, (cs2, csc2) in enumerate(csdict.items()):
                    if j < i:
                        continue

                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    chiinv[block1, block2][idx1, idx2] = invert[:, i, j]
                    chiinv[block2, block1][idx2, idx1] = invert[:, i, j]

            if logdet:
                return chiinv, ld
            else:
                return chiinv
        else:
            return [None if chivec is None else chivec.inv(logdet) for chivec in chivecs]

    def get_chiinv_sparse(self, params, logdet=False):
        chi = self.get_chi(params)

        if isinstance(chi, list):
            return [None if chivec is None else chivec.inv(logdet) for chivec in chi]
        else:
            chisparse = sps.csc_matrix(chi)
            cf = cholesky(chisparse)

            if logdet:
                return (cf.inv(), cf.logdet())
            else:
                return cf.inv()

    def get_chiinv_byfreq_cliques(self, params, logdet=False, cholesky=False):
        chi = self.get_chi(params, cliques=True)

        if isinstance(chi, list):
            return [None if chivec is None else chivec.inv(logdet) for chivec in chi]
        else:
            ld = 0

            # first invert all the cliques
            for clcount in range(self._clcount):
                idx = self._cliques == clcount

                if np.any(idx):
                    idx2 = np.ix_(idx, idx)

                    if cholesky:
                        cf = sl.cho_factor(chi[idx2])

                        if logdet:
                            ld += 2.0 * np.sum(np.log(np.diag(cf[0])))

                        chi[idx2] = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                    else:
                        chi2 = chi[idx2]

                        if logdet:
                            ld += np.linalg.slogdet(chi2)[1]

                        chi[idx2] = np.linalg.inv(chi2)

            # then do the pure diagonal terms
            idx = self._cliques == -1

            if logdet:
                ld += np.sum(np.log(chi[idx, idx]))

            chi[idx, idx] = 1.0 / chi[idx, idx]

            return (chi, ld) if logdet else chi

    def get_phiinv_byfreq_cliques(self, params, logdet=False, cholesky=False):
        phi = self.get_phi(params, cliques=True)

        if isinstance(phi, list):
            return [None if phivec is None else phivec.inv(logdet) for phivec in phi]
        else:
            ld = 0

            # first invert all the cliques
            for clcount in range(self._clcount):
                idx = self._cliques == clcount

                if np.any(idx):
                    idx2 = np.ix_(idx, idx)

                    if cholesky:
                        cf = sl.cho_factor(phi[idx2])

                        if logdet:
                            ld += 2.0 * np.sum(np.log(np.diag(cf[0])))

                        phi[idx2] = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                    else:
                        phi2 = phi[idx2]

                        if logdet:
                            ld += np.linalg.slogdet(phi2)[1]

                        phi[idx2] = np.linalg.inv(phi2)

            # then do the pure diagonal terms
            idx = self._cliques == -1

            if logdet:
                ld += np.sum(np.log(phi[idx, idx]))

            phi[idx, idx] = 1.0 / phi[idx, idx]

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
    def _setcliques(self, slices, csdict):
        # each column in idxmatrix (mind the .T) corresponds to the indices
        # that participate in a common signal for a given pulsar
        idxmatrix = np.array([csc._idx[cs] for cs, csc in csdict.items()]).T

        # each row in the updated idxmatrix corresponds to a set of "global"
        # Phi indices that are correlated across pulsars
        idxmatrix = idxmatrix + np.array([slices[csc].start for cs, csc in csdict.items()])

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
                    self._cliques[np.in1d(self._cliques, allidx)] = self._clcount

                self._clcount = self._clcount + 1
            else:
                # if we find at least one clique, assign all indices in idx
                # to the maximum clique index

                self._cliques[idxs] = maxidx

                # since cliques are "contagious", reassign all the other
                # clique indices that we found to maxidx
                if len(allidx) > 1:
                    self._cliques[np.in1d(self._cliques, allidx)] = maxidx

    # add cliques from individual pulsar phis; these will never overlap
    # TO DO: at this point Phi could be defined as a smarter KernelMatrix!
    def _setpulsarcliques(self, slices, phis):
        for sc, phi in zip(self._signalcollections, phis):
            if phi is not None:
                for clindex in range(getattr(phi, "_clcount", 0)):
                    phiind = np.where(phi._cliques == clindex)[0]

                    if len(phiind) > 0:
                        try:
                            self._cliques[slices[sc].start + phiind] = self._clcount
                            self._clcount = self._clcount + 1
                        except Exception:  # pragma: no cover
                            logger.exception("Exception raised in computing cliques")
                            logger.info(self._cliques.shape)
                            logger.info("phiind", phiind, len(phiind))
                            logger.info(slices)
                            raise

    def get_phi(self, params, cliques=False):
        phis = [signalcollection.get_phi(params) for signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            if np.any([phi.ndim == 2 for phi in phis if phi is not None]):
                # if we have any dense matrices,
                Phi = sl.block_diag(*[np.diag(phi) if phi.ndim == 1 else phi for phi in phis if phi is not None])
            else:
                Phi = np.diag(np.concatenate([phi for phi in phis if phi is not None]))

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
                pairs = itertools.combinations(csdict.items(), 2)

                for (cs1, csc1), (cs2, csc2) in pairs:
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    if crossdiag.ndim == 1:
                        Phi[block1, block2][idx1, idx2] += crossdiag
                        Phi[block2, block1][idx2, idx1] += crossdiag
                    else:
                        Phi[block1, block2][np.ix_(idx1, idx2)] += crossdiag
                        Phi[block2, block1][np.ix_(idx2, idx1)] += crossdiag

            return Phi
        else:
            return phis

    def get_chi(self, params, cliques=False):
        chis = [signalcollection.get_chi(params) for signalcollection in self._signalcollections]

        # if we found common signals, we'll return a big phivec matrix,
        # otherwise a list of phivec vectors (some of which possibly None)
        if self._commonsignals:
            if np.any([chi.ndim == 2 for chi in chis if chi is not None]):
                # if we have any dense matrices,
                Chi = sl.block_diag(*[np.diag(chi) if chi.ndim == 1 else chi for chi in chis if chi is not None])
            else:
                Chi = np.diag(np.concatenate([chi for chi in chis if chi is not None]))

            # get a dictionary of slices locating each pulsar in Phi matrix
            slices = self._get_slices(chis)

            # self._cliques is a vector of the same size as the Phi matrix
            # for each Phi index i, self._cliques[i] is -1 if row/column
            # belong to no clique, or it gives the clique number otherwise
            if cliques:
                self._resetcliques(Chi.shape[0])
                self._setpulsarcliques(slices, chis)

            # iterate over all common signal classes
            for csclass, csdict in self._commonsignals.items():
                # first figure out which indices are used in this common signal
                # and update the clique index
                if cliques:
                    self._setcliques(slices, csdict)

                # now iterate over all pairs of common signal instances
                pairs = itertools.combinations(csdict.items(), 2)

                for (cs1, csc1), (cs2, csc2) in pairs:
                    crossdiag = csclass.get_phicross(cs1, cs2, params)

                    block1, idx1 = slices[csc1], csc1._idx[cs1]
                    block2, idx2 = slices[csc2], csc2._idx[cs2]

                    if crossdiag.ndim == 1:
                        Chi[block1, block2][idx1, idx2] += crossdiag
                        Chi[block2, block1][idx2, idx1] += crossdiag
                    else:
                        Chi[block1, block2][np.ix_(idx1, idx2)] += crossdiag
                        Chi[block2, block1][np.ix_(idx2, idx1)] += crossdiag

            return Chi
        else:
            return chis

    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct : ct + n] if n > 1 else float(xs[ct])
            ct += n
        return ret

    def get_lnprior(self, params):
        # map parameter vector if needed
        params = params if isinstance(params, dict) else self.map_params(params)

        return np.sum([p.get_logpdf(params=params) for p in self.params])

    @property
    def pulsars(self):
        return [p.psrname for p in self._signalcollections]

    def _set_signal_dict(self):
        """ Set signal dictionary"""

        self._signal_dict = {}
        sig_list = []
        for ct1, sc in enumerate(self._signalcollections):
            for ct2, sig in enumerate(sc._signals):
                if sig.name not in sig_list:
                    sig_list.append(sig.name)
                    self._signal_dict[sig.name] = sig
                else:
                    msg = "Duplicate signal {} from objects {} and {}."
                    msg += "\nThis functionality was added in v1.1.0 and may"
                    msg += " cause post v1.1.0 functionality to break."
                    msg += "\nThis may not cause other errors but it is"
                    msg += " recommended that you use a custom name for one"
                    msg += " of the duplicate signals.\n"
                    logger.warn(msg.format(sig.name, sig, self._signal_dict[sig.name]))

    @property
    def signals(self):
        """ Return signal dictionary."""
        return self._signal_dict

    def get_signal(self, name):
        """Returns ``Signal`` instance given the signal name."""
        return self._signal_dict[name]

    def summary(self, include_params=True, to_stdout=False):
        """generate summary string for PTA model

        :param include_params: [bool]
            list all parameters for each signal
        :param to_stdout: [bool]
            print summary to `stdout` instead of returning it
        :return: [string]
        """
        summary = "enterprise v" + __version__ + ",  "
        summary += "Python v" + _py_version + "\n"
        summary += "=" * 90 + "\n"
        summary += "\n"
        row = ["Signal Name", "Signal Class", "no. Parameters"]
        summary += "{: <40} {: <30} {: <20}\n".format(*row)
        summary += "=" * 90 + "\n"
        cpcount, copcount = 0, 0
        for sc in self._signalcollections:
            for sig in sc._signals:
                for p in sig.param_names:
                    if sc.psrname not in p:
                        cpcount += 1
                row = [sig.name, sig.__class__.__name__, len(sig.param_names)]
                summary += "{: <40} {: <30} {: <20}\n".format(*row)
                if include_params:
                    summary += "\n"
                    summary += "params:\n"
                    for par in sig._params.values():
                        if isinstance(par, ConstantParameter):
                            copcount += 1
                        summary += "{!s: <90}\n".format(par.__repr__())
                summary += "_" * 90 + "\n"
        summary += "=" * 90 + "\n"
        summary += "Total params: {}\n".format(len(self.param_names) + copcount)
        summary += "Varying params: {}\n".format(len(self.param_names))
        summary += "Common params: {}\n".format(cpcount)
        summary += "Fixed params: {}\n".format(copcount)
        summary += "Number of pulsars: {}\n".format(len(self._signalcollections))
        if to_stdout:
            logger.info(summary)
        else:
            return summary


def SignalCollection(metasignals):  # noqa: C901
    """Class factory for ``SignalCollection`` objects."""

    @six.add_metaclass(MetaCollection)
    class SignalCollection(object):
        _metasignals = metasignals

        def __init__(self, psr):
            self.psrname = psr.name
            # instantiate all the signals with a pulsar
            self._signals = [metasignal(psr) for metasignal in self._metasignals]

            self._residuals = psr.residuals

            self._set_cache_parameters()

        def __add__(self, other):
            return PTA([self, other])

        # TODO: this could be implemented more cleanly
        def _set_cache_parameters(self):
            """ Sets the cache for various signal types."""

            self.white_params = []
            self.basis_params = []
            self.delay_params = []
            for signal in self._signals:
                if signal.signal_type == "white noise":
                    self.white_params.extend(signal.ndiag_params)
                elif signal.signal_type in ["basis", "common basis"]:
                    # to support GP coefficients, and yet do the right thing
                    # for common GPs, which do not have coefficients yet
                    self.delay_params.extend(getattr(signal, "delay_params", []))
                    self.basis_params.extend(signal.basis_params)
                elif signal.signal_type in ["deterministic"]:
                    self.delay_params.extend(signal.delay_params)
                else:
                    msg = "{} signal type not recognized! Caching ".format(signal.signal_type)
                    msg += "may not work correctly for this signal."
                    logger.error(msg)

        def cache_clear(self):
            for instance in [self] + self.signals:
                kill = [attr for attr in instance.__dict__ if attr.startswith("_cache")]

                for attr in kill:
                    del instance.__dict__[attr]

        # a candidate for memoization
        @property
        def params(self):
            return sorted({param for signal in self._signals for param in signal.params}, key=lambda par: par.name)

        @property
        def param_names(self):
            ret = []
            for p in self.params:
                if p.size:
                    for ii in range(0, p.size):
                        ret.append(p.name + "_{}".format(ii))
                else:
                    ret.append(p.name)
            return ret

        @property
        def signals(self):
            return self._signals

        # Timing models signals go in M, other signals in F
        @property
        def signals_M(self):
            return [sig for sig in self.signals if sig.is_timing_model]

        @property
        def signals_F(self):
            return [sig for sig in self.signals if not sig.is_timing_model]

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

            idx, hashlist, cc, nrow = {}, [], 0, None
            for signal in signals:
                Fmat = signal.get_basis()

                if Fmat is not None:
                    nrow = Fmat.shape[0]

                    if not signal.basis_params:
                        idx[signal] = []

                        for i, column in enumerate(Fmat.T):
                            colhash = hash(column.tobytes())

                            if signal.basis_combine and colhash in hashlist:
                                # if we're combining the basis for this signal
                                # and we have seen this column already, make a note
                                # of where it was

                                j = hashlist.index(colhash)
                                idx[signal].append(j)
                            else:
                                # if we're not combining or we haven't seen it already
                                # save the hash and make a note it's new

                                hashlist.append(colhash)
                                idx[signal].append(cc)
                                cc += 1
                    elif signal.basis_params:
                        nf = Fmat.shape[1]
                        idx[signal] = list(range(cc, cc + nf))
                        cc += nf

            if not idx:
                return {}, None
            else:
                ncol = len(np.unique(sum(idx.values(), [])))
                return ({key: np.array(idx[key]) for key in idx.keys()}, np.zeros((nrow, ncol)))

        # goofy way to cache _idx
        def __getattr__(self, par):
            if par in ("_idx", "_Fmat"):
                self._idx, self._Fmat = self._combine_basis_columns(self._signals)
            elif par in ("_idx_M", "_Fmat_M"):
                # Timing model signals are used to make the D matrix
                self._idx_M, self._Fmat_M = self._combine_basis_columns(self.signals_M)
            elif par in ("_idx_F", "_Fmat_F"):
                # Other signals are used to make C from D
                self._idx_F, self._Fmat_F = self._combine_basis_columns(self.signals_F)
            else:
                raise AttributeError("{} object has no attribute {}".format(self.__class__, par))
            return getattr(self, par)

        @cache_call("white_params")
        def get_ndiag(self, params):
            ndiags = [signal.get_ndiag(params) for signal in self._signals]
            return sum(ndiag for ndiag in ndiags if ndiag is not None)

        @cache_call("delay_params")
        def get_delay(self, params):
            delays = [signal.get_delay(params) for signal in self._signals]
            return sum(delay for delay in delays if delay is not None)

        @cache_call("delay_params")
        def get_detres(self, params):
            return self._residuals - self.get_delay(params)

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @cache_call("basis_params", limit=1)
        def get_basis(self, params={}):
            for signal in self._signals:
                if signal in self._idx:
                    self._Fmat[:, self._idx[signal]] = signal.get_basis(params)
            return self._Fmat

        # This is the M matrix.
        # By definition there cannot be parameters here, but we take the params
        # argument anyway to make cache_call happy
        @cache_call([])
        def get_basis_M(self, params={}):
            for signal in self.signals_M:
                if signal in self._idx_M:
                    self._Fmat_M[:, self._idx_M[signal]] = signal.get_basis({})
            return self._Fmat_M

        # this is the F matrix that has the basis for the signals that depend on parameters
        @cache_call("basis_params", limit=1)
        def get_basis_F(self, params={}):
            for signal in self.signals_F:
                if signal in self._idx:
                    self._Fmat_F[:, self._idx_F[signal]] = signal.get_basis(params)
            return self._Fmat_F

        def get_phiinv(self, params):
            return self.get_phi(params).inv()

        # returns a KernelMatrix object
        def get_phi(self, params):
            if self._Fmat is None:
                return None

            phi = KernelMatrix(self._Fmat.shape[1])

            for signal in self._signals:
                if signal in self._idx:
                    phi = phi.add(signal.get_phi(params), self._idx[signal])

            return phi

        # returns a KernelMatrix object
        def get_chi(self, params):
            """Like phi, but only for signals that depend on parameters"""

            F = self.get_basis_F(params)
            if F is not None:  # Anything to do?

                chi = KernelMatrix(F.shape[1])

                for signal in self.signals_F:
                    if signal in self._idx_F:
                        chi = chi.add(signal.get_phi(params), self._idx_F[signal])

                return chi
            else:
                return None  # If no non-timing-model signals with bases, then no F or chi

        def get_chiinv(self, params):
            chi = self.get_chi(params)
            if chi:
                return chi.inv()
            else:
                return None

        @cache_call(["basis_params", "white_params", "delay_params"])
        def get_TNr(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=T)

        @cache_call(["white_params", "delay_params"])
        def get_MNr(self, params):
            M = self.get_basis_M(params)
            if M is None:
                return None
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=M)

        @cache_call(["basis_params", "white_params", "delay_params"])
        def get_FNr(self, params):
            F = self.get_basis_F(params)
            if F is None:
                return None
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=F)

        @cache_call(["basis_params", "white_params"])
        def get_TNT(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            return Nvec.solve(T, left_array=T)

        # Return M^T N^-1 M in csc form
        @cache_call("white_params")
        def get_MNM(self, params):
            M = self.get_basis_M(params)
            if M is None:
                return None
            Nvec = self.get_ndiag(params)
            # Could this be done in a way that gives a sparse matrix to start with?
            return sps.csc_matrix(Nvec.solve(M, left_array=M))

        @cache_call("white_params")
        def get_MNM_cholesky(self, params):
            MNM = self.get_MNM(params)
            if MNM is None:
                return None
            return cholesky(MNM)

        @cache_call("white_params")
        def get_MNM_logdet(self, params):
            return self.get_MNM_cholesky(params).logdet()

        @cache_call(["basis_params", "white_params"])
        def get_FNF(self, params):
            F = self.get_basis_F(params)
            if F is None:
                return None
            Nvec = self.get_ndiag(params)
            return Nvec.solve(F, left_array=F)  # F^T N^{-1} F

        @cache_call(["basis_params", "white_params"])
        def get_MNF(self, params):
            M = self.get_basis_M(params)
            F = self.get_basis_F(params)
            if F is None or M is None:
                return None
            Nvec = self.get_ndiag(params)
            return Nvec.solve(F, left_array=M)  # M^T N^{-1} F

        @cache_call(["basis_params", "white_params"])
        def get_MNMMNF(self, params):
            cf = self.get_MNM_cholesky(params)
            MNF = self.get_MNF(params)
            if cf is None or MNF is None:
                return None
            return cf(MNF)  # (MNM)^-1 MNF

        # Returns r^T N r and ln(det(N))
        @cache_call(["white_params", "delay_params"])
        def get_rNr_logdet(self, params):
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=res, logdet=True)

        @cache_call(["basis_params", "white_params"])
        def get_FDF(self, params):
            MNMMNF = self.get_MNMMNF(params)
            MNF = self.get_MNF(params)
            if MNF is None or MNMMNF is None:
                return self.get_FNF(params)
            return self.get_FNF(params) - np.tensordot(MNF, MNMMNF, (0, 0))  # FNF - MNF^T MNM^-1 MNF

        @cache_call(["basis_params", "white_params", "delay_params"])
        def get_FDr(self, params):
            MNMMNF = self.get_MNMMNF(params)
            MNr = self.get_MNr(params)
            if MNr is None or MNMMNF is None:
                return self.get_FNr(params)
            return self.get_FNr(params) - np.tensordot(MNMMNF, MNr, (0, 0))  # FNr - MNF^T MNM^-1 MNr

        # Returns r^T D^-1 r and ln(det(D))
        @cache_call(["white_params", "delay_params"])
        def get_rDr_logdet(self, params):
            M = self.get_basis_M(params)
            rNr, logdet_N = self.get_rNr_logdet(params)
            if M is None:  # No model parameters so D=N
                return (rNr, logdet_N)
            MNr = self.get_MNr(params)
            cf = self.get_MNM_cholesky(params)
            return (rNr - np.dot(MNr, cf(MNr)), logdet_N + self.get_MNM_logdet(params))

        # TO DO: cache how?
        def get_logsignalprior(self, params):
            return sum(signal.get_logsignalprior(params) for signal in self._signals)

    return SignalCollection


def cache_call(attrs, limit=2):
    """This decorator caches the output of a class method that takes
    a single parameter 'params'. It saves the cache in the instance
    attributes _cache_<methodname> and _cache_list_<methodname>.

    The cache keys are listed in the class attribute (or attributes)
    specified in the initial decorator call. For instance, if
    the decorator is applied as @cache_call('basis_params'), then
    the parameters listed in self.basis_params (together with their values)
    will be used as the key.

    The parameter 'limit' specifies the number of entries saved
    in the cache."""

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
            # key = tuple([(key, params[key]) for key in keys if key in params])

            # make sure the cache is part of the object
            if not hasattr(self, "_cache_" + func.__name__):
                msg = "Create cache {} for signal {}".format(func.__name__, self.__class__)
                logger.debug(msg)

                setattr(self, "_cache_" + func.__name__, {})
                setattr(self, "_cache_list_" + func.__name__, [])

            cache = getattr(self, "_cache_" + func.__name__)
            cache_list = getattr(self, "_cache_list_" + func.__name__)

            if key not in cache:
                msg = "Setting cache for {} in {}: {}".format(attrs, self.__class__, key)
                logger.debug(msg)

                cache_list.append(key)
                cache[key] = func(self, params)

                if len(cache_list) > limit:
                    _ = cache.pop(cache_list.pop(0), None)  # noqa: F841
            else:
                msg = "Retrieving cache for {} in {}: {}".format(attrs, self.__class__, key)
                logger.debug(msg)

            return cache[key]

        return wrapper

    return cache_decorator


class csc_matrix_alt(sps.csc_matrix):
    """Sub-class of ``scipy.sparse.csc_matrix`` with custom ``add`` and
    ``solve`` methods.
    """

    def _add_diag(self, other):
        other_diag = sps.dia_matrix((other, np.array([0])), shape=(other.shape[0], other.shape[0]))
        return self._binopt(other_diag, "_plus_")

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
            mult = np.array(other / self[:, None])
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
            s2 = set(np.concatenate([np.arange(len(nvec))[slc] for slc in slices]))
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
            ZNXr = np.dot(Z[self._idx, :].T, X[self._idx, :] / self._nvec[self._idx, None])
        else:
            ZNXr = 0
        for slc, block in zip(self._slices, self._blocks):
            Zblock = Z[slc, :]
            Xblock = X[slc, :]

            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block + np.diag(self._nvec[slc]))
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

        NX = X / self._nvec[:, None]
        for slc, block in zip(self._slices, self._blocks):
            Xblock = X[slc, :]
            if slc.stop - slc.start > 1:
                cf = sl.cho_factor(block + np.diag(self._nvec[slc]))
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
                cf = sl.cho_factor(block + np.diag(self._nvec[slc]))
                logdet += np.sum(2 * np.log(np.diag(cf[0])))
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
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
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
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
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
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                ZNX -= beta * np.outer(zn.T, xn)
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logdet = np.einsum("i->", np.log(self._nvec))
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
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
