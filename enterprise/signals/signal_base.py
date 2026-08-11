# signal_base.py
"""
Defines the signal base classes and metaclasses. All signals will then be
derived from these base classes.
"""

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
from enterprise.signals.utils import indices_from_slice

from enterprise import __version__
from sys import version

_py_version = version.split(" ")[0]

# logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _simplememobyid_keycheck(key, arg):
    if isinstance(key, Sequence):
        return isinstance(arg, Sequence) and len(key) == len(arg) and all(e1 is e2 for e1, e2 in zip(key, arg))
    else:
        return key is arg


def simplememobyid(method):
    """This decorator caches the last call of a class method that takes
    a single parameter `arg`. It holds a reference to the last `arg` as `key`,
    and uses the cached value if `arg is key`. If `arg` is a Sequence,
    then the decorator uses the cached value if the `is` relation is true
    element by element."""

    def memoizedfunc(self, arg):
        cacheloc = "_memo" + method.__name__

        # if not hasattr(self, cacheloc) or self.__dict__[cacheloc][0] is not arg:
        if not hasattr(self, cacheloc) or not _simplememobyid_keycheck(self.__dict__[cacheloc][0], arg):
            self.__dict__[cacheloc] = (arg, method(self, arg))

        return self.__dict__[cacheloc][1]

    return memoizedfunc


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
            if p.size is not None:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret

    def __repr__(self):
        return "<Enterprise Signal object " + self.signal_id + "[" + ", ".join(p.name for p in self.params) + "]>"

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


def LogLikelihoodDenseCholesky(pta):
    return LogLikelihood(pta, cholesky_sparse=False)


class LogLikelihood(object):
    def __init__(self, pta, cholesky_sparse=True):
        self.pta = pta
        self.cholesky_sparse = cholesky_sparse

    @simplememobyid
    def _block_TNT(self, TNTs):
        if self.cholesky_sparse:
            return sps.block_diag(TNTs, "csc")
        else:
            return sl.block_diag(*TNTs)

    @simplememobyid
    def _block_TNr(self, TNrs):
        return np.concatenate(TNrs)

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

        # Add factors of log(2pi) for the likelihood normalization
        ntot = sum(sc._residuals.size for sc in self.pta._signalcollections)
        loglike -= 0.5 * ntot * np.log(2 * np.pi)

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))

        # red noise piece
        if self.pta._commonsignals:
            phiinv, logdet_phi = phiinvs

            TNT = self._block_TNT(TNTs)
            TNr = self._block_TNr(TNrs)

            if self.cholesky_sparse:
                try:
                    Sigma_sp = TNT + sps.csc_matrix(phiinv)

                    if hasattr(self, "cf_sp"):
                        # Have analytical decomposition already. Just do update
                        self.cf_sp.cholesky_inplace(Sigma_sp)
                    else:
                        # Do analytical and numerical Sparse Cholesky
                        self.cf_sp = cholesky(Sigma_sp)

                    expval = self.cf_sp(TNr)
                    logdet_sigma = self.cf_sp.logdet()
                except CholmodError:  # pragma: no cover
                    return -np.inf
            else:
                try:
                    cf = sl.cho_factor(TNT + phiinv)  # cf(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                    logdet_sigma = 2 * np.sum(np.log(np.diag(cf[0])))
                except sl.LinAlgError:  # pragma: no cover
                    return -np.inf

            loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)
        else:
            for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
                if TNr is None:
                    continue

                phiinv, logdet_phi = pl
                Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                except sl.LinAlgError:  # pragma: no cover
                    return -np.inf

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))

                loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike


class _HouseholderUnavailable(NotImplementedError):
    """Raised when the Schur likelihood cannot be evaluated."""


class SchurLogLikelihood(LogLikelihood):
    """Log-likelihood using a Schur complement decomposition.

    Basis coefficients for individual pulsars are eliminated before the
    matrix for common signals is factorized. Unsupported models are evaluated
    with :class:`LogLikelihood`.
    """

    @staticmethod
    def _validate_split(signalcollection, params, common_idx, pulsar_idx):
        """Check for dense priors that mix common and individual columns."""

        common = set(common_idx)
        pulsar = set(pulsar_idx)
        if not common or not pulsar:
            return

        for signal, columns in signalcollection._idx.items():
            columns = set(columns)
            if columns & common and columns & pulsar:
                phi = signal.get_phi(params)
                if phi is not None and np.ndim(phi) == 2:
                    raise _HouseholderUnavailable(
                        "Dense prior for signal {!r} in pulsar {!r} includes "
                        "both common and individual basis columns.".format(signal.signal_id, signalcollection.psrname)
                    )

    @staticmethod
    def _sqrtsolve(noise, array):
        """Apply a square-root solve for the white-noise covariance."""

        if not getattr(noise, "_has_sqrtsolve", hasattr(noise, "sqrtsolve")):
            raise NotImplementedError("{} does not implement sqrtsolve.".format(noise.__class__.__name__))
        try:
            return noise.sqrtsolve(array)
        except NotImplementedError as error:
            raise NotImplementedError("{} does not implement sqrtsolve.".format(noise.__class__.__name__)) from error

    @classmethod
    def _get_data(cls, signalcollection, params, common_idx, pulsar_idx):
        """Return the whitened residual, basis, and QR factors for one pulsar."""

        try:
            noise = signalcollection.get_ndiag(params)
            residual = cls._sqrtsolve(noise, signalcollection.get_detres(params))

            local_basis = signalcollection.get_basis_pulsar_only(params, pulsar_idx)
            if local_basis is not None:
                local_basis = cls._sqrtsolve(noise, local_basis)

            common_basis = signalcollection.get_basis_common(params, common_idx)
            if common_basis is None:
                qr_raw = tau = R = None
            else:
                common_basis = cls._sqrtsolve(noise, common_basis)
                (qr_raw, tau), R = sl.qr(common_basis, mode="raw", check_finite=False)
        except NotImplementedError as error:
            raise _HouseholderUnavailable(str(error)) from error
        return residual, local_basis, qr_raw, tau, R

    @staticmethod
    def _build_local_prior_root(phiinv, size):
        if phiinv is None:
            return np.zeros((size, size))
        if np.ndim(phiinv) == 1:
            return np.diag(np.sqrt(np.maximum(np.asarray(phiinv), 0.0)))
        return sl.cholesky(np.asarray(phiinv), lower=False, check_finite=False)

    @staticmethod
    def _apply_qt_householder(qr_raw, tau, array):
        """Apply the transpose of Q from a raw QR decomposition."""

        matrix = array[:, np.newaxis] if array.ndim == 1 else array
        ormqr = sl.lapack.get_lapack_funcs("ormqr", (qr_raw, matrix))
        transformed, work, info = ormqr("L", "T", qr_raw, tau, matrix, lwork=-1, overwrite_c=0)
        if info != 0:  # pragma: no cover
            raise np.linalg.LinAlgError("LAPACK ormqr workspace query failed (info={}).".format(info))
        transformed, _, info = ormqr(
            "L",
            "T",
            qr_raw,
            tau,
            matrix,
            lwork=int(work[0].real),
            overwrite_c=0,
        )
        if info != 0:  # pragma: no cover
            raise np.linalg.LinAlgError("LAPACK ormqr failed (info={}).".format(info))
        return transformed[:, 0] if array.ndim == 1 else transformed

    @staticmethod
    def _solve_augmented_rrqr(matrix, data, common_local=None, rtol=1e-12):
        """Solve the augmented system using QR with column pivoting."""

        Q, R, pivots = sl.qr(matrix, mode="economic", pivoting=True, check_finite=False)
        transformed = np.dot(Q.T, data)
        diagonal = np.abs(np.diag(R))
        rank = int(np.sum(diagonal > float(diagonal[0]) * float(rtol))) if diagonal.size else 0

        permuted = np.zeros(R.shape[1])
        logdet = 0.0
        leading = None
        if rank:
            leading = R[:rank, :rank]
            permuted[:rank] = sl.solve_triangular(leading, transformed[:rank], check_finite=False)
            logdet = 2.0 * np.sum(np.log(diagonal[:rank]))

        solution = np.zeros(R.shape[1])
        solution[pivots] = permuted
        residual = data - np.dot(matrix, solution)
        quadratic = np.dot(residual, residual)

        delta = None
        if common_local is not None:
            if not rank:
                delta = np.zeros((common_local.shape[0], common_local.shape[0]))
            else:
                permuted_common = common_local.T[pivots, :]
                solved = sl.solve_triangular(
                    leading,
                    permuted_common[:rank, :],
                    trans="T",
                    check_finite=False,
                )
                delta = np.dot(solved.T, solved)
                delta = 0.5 * (delta + delta.T)
        return solution, quadratic, float(logdet), delta

    def _householder_likelihood(self, params):
        signalcollections = self.pta._signalcollections
        basis_splits = {
            signalcollection: self.pta._get_basis_split(signalcollection) for signalcollection in signalcollections
        }
        for signalcollection in signalcollections:
            common_idx, pulsar_idx = basis_splits[signalcollection]
            self._validate_split(signalcollection, params, common_idx, pulsar_idx)

        common_phi = self.pta.get_phi_common(params)
        if common_phi is None:
            raise _HouseholderUnavailable("The model has no common basis columns.")

        common_offsets, offset = {}, 0
        for signalcollection in signalcollections:
            size = basis_splits[signalcollection][0].size
            if size:
                common_offsets[signalcollection] = slice(offset, offset + size)
                offset += size

        priors = []
        for signalcollection in signalcollections:
            pulsar_idx = basis_splits[signalcollection][1]
            phiinv, logdet = signalcollection.get_phiinv_pulsar_only(params, pulsar_idx, logdet=True)
            priors.append((phiinv, float(logdet)))
        logdet_white = sum(sum(value[1:]) for value in self.pta.get_rNr_logdet(params))

        # whiten the residuals and eliminate coefficients for individual pulsars
        z_values, R_values, deltas, block_collections = [], [], [], []
        local_quadratic = local_logdet = local_phi_logdet = 0.0
        for signalcollection, (local_phiinv, local_phi_logdet_i) in zip(signalcollections, priors):
            common_idx, pulsar_idx = basis_splits[signalcollection]
            residual, local_basis, qr_raw, tau, R = self._get_data(signalcollection, params, common_idx, pulsar_idx)

            if qr_raw is None:
                if local_basis is None:
                    local_quadratic += np.dot(residual, residual)
                    continue
                root = self._build_local_prior_root(local_phiinv, local_basis.shape[1])
                matrix = np.vstack([local_basis, root])
                data = np.concatenate([residual, np.zeros(local_basis.shape[1])])
                _, quadratic, logdet, _ = self._solve_augmented_rrqr(matrix, data)
                local_quadratic += quadratic
                local_logdet += logdet
                local_phi_logdet += local_phi_logdet_i
                continue

            common_size = R.shape[0]
            transformed_residual = self._apply_qt_householder(qr_raw, tau, residual)
            residual_common = transformed_residual[:common_size]
            residual_local = transformed_residual[common_size:]

            if local_basis is None:
                z_value = residual_common.copy()
                delta = np.zeros((common_size, common_size))
                local_quadratic += np.dot(residual_local, residual_local)
            else:
                transformed_local = self._apply_qt_householder(qr_raw, tau, local_basis)
                common_local = transformed_local[:common_size, :]
                local_local = transformed_local[common_size:, :]
                root = self._build_local_prior_root(local_phiinv, local_local.shape[1])
                matrix = np.vstack([local_local, root])
                data = np.concatenate([residual_local, np.zeros(local_local.shape[1])])
                solution, quadratic, logdet, delta = self._solve_augmented_rrqr(matrix, data, common_local=common_local)
                z_value = residual_common - np.dot(common_local, solution)
                local_quadratic += quadratic
                local_logdet += logdet
                local_phi_logdet += local_phi_logdet_i

            z_values.append(z_value)
            R_values.append(R)
            deltas.append(delta)
            block_collections.append(signalcollection)

        # assemble the matrix for common signals
        common_dimension = sum(R.shape[0] for R in R_values)
        if not common_dimension:
            raise _HouseholderUnavailable("The model has no factorizable common basis.")

        sigma = np.eye(common_dimension)
        schur_offsets, offset = [], 0
        for R, delta in zip(R_values, deltas):
            block = slice(offset, offset + R.shape[0])
            schur_offsets.append(block)
            sigma[block, block] += delta
            offset = block.stop

        for index1, (collection1, R1, block1) in enumerate(zip(block_collections, R_values, schur_offsets)):
            phi_block1 = common_offsets[collection1]
            for index2 in range(index1, len(block_collections)):
                collection2 = block_collections[index2]
                R2, block2 = R_values[index2], schur_offsets[index2]
                phi_block2 = common_offsets[collection2]
                phi = common_phi[phi_block1, phi_block2]
                block = np.dot(np.dot(R1, phi), R2.T)
                sigma[block1, block2] += block
                if index1 != index2:
                    sigma[block2, block1] += block.T

        # factor the remaining matrix and combine the likelihood terms
        factor = sl.cho_factor(sigma, lower=True, check_finite=False)
        z = np.concatenate(z_values)
        common_quadratic = np.dot(z, sl.cho_solve(factor, z, check_finite=False))
        common_logdet = 2.0 * np.sum(np.log(np.diag(factor[0])))

        total_toas = sum(signalcollection._residuals.size for signalcollection in signalcollections)
        loglike = -0.5 * (
            local_quadratic
            + common_quadratic
            + common_logdet
            + local_logdet
            + local_phi_logdet
            + logdet_white
            + total_toas * np.log(2 * np.pi)
        )
        loglike += sum(self.pta.get_logsignalprior(params))
        return float(loglike)

    def __call__(self, xs, phiinv_method="cliques"):
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)
        if not self.pta._commonsignals:
            return super(SchurLogLikelihood, self).__call__(params, phiinv_method=phiinv_method)

        try:
            return self._householder_likelihood(params)
        except _HouseholderUnavailable:
            return super(SchurLogLikelihood, self).__call__(params, phiinv_method=phiinv_method)
        except sl.LinAlgError:
            return -np.inf


class PTA(object):
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
            if p.size is not None:
                for ii in range(0, p.size):
                    ret.append(p.name + "_{}".format(ii))
            else:
                ret.append(p.name)
        return ret

    @property
    def pulsarmodels(self):
        return self._signalcollections

    def __repr__(self):
        return "<Enterprise PTA object: " + ", ".join(self.keys()) + ">"

    # emulate a dictionary

    def __len__(self):
        return len(self._signalcollections)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._signalcollections[key]
        else:
            for sc in self._signalcollections:
                if sc.psrname == key:
                    return sc

            raise KeyError("Pulsar model not found")

    def keys(self):
        return [sc.psrname for sc in self._signalcollections]

    def values(self):
        return self._signalcollections

    def items(self):
        return [(sc.psrname, sc) for sc in self._signalcollections]

    # tensor quantities assembled from individual pulsar models

    def get_TNr(self, params):
        return [signalcollection.get_TNr(params) for signalcollection in self._signalcollections]

    def get_TNT(self, params):
        return [signalcollection.get_TNT(params) for signalcollection in self._signalcollections]

    def get_rNr_logdet(self, params):
        return [signalcollection.get_rNr_logdet(params) for signalcollection in self._signalcollections]

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

    def _get_basis_split(self, signalcollection):
        """Return common and individual basis indices for one pulsar model."""

        common_signals = frozenset(
            signal
            for signals in self._commonsignals.values()
            for signal, collection in signals.items()
            if collection is signalcollection
        )
        return signalcollection._compute_basis_split(common_signals)

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

    def get_phi_common(self, params):
        """Return the covariance for basis columns of common signals."""

        if not self._commonsignals:
            return None

        common_indices = {
            signalcollection: self._get_basis_split(signalcollection)[0] for signalcollection in self._signalcollections
        }
        slices, positions, offset = {}, {}, 0
        for signalcollection in self._signalcollections:
            common_idx = common_indices[signalcollection]
            if common_idx.size:
                slices[signalcollection] = slice(offset, offset + common_idx.size)
                positions[signalcollection] = {int(column): index for index, column in enumerate(common_idx)}
                offset += common_idx.size

        if not offset:
            return None

        phi_common = np.zeros((offset, offset))
        for signalcollection, block in slices.items():
            common_idx = common_indices[signalcollection]
            phi = np.asarray(signalcollection.get_phi_common(params, common_idx))
            phi_common[block, block] = np.diag(phi) if phi.ndim == 1 else phi

        for common_class, common_signals in self._commonsignals.items():
            for (signal1, collection1), (signal2, collection2) in itertools.combinations(common_signals.items(), 2):
                rows = slices[collection1].start + np.asarray(
                    [positions[collection1][int(column)] for column in collection1._idx[signal1]],
                    dtype=int,
                )
                columns = slices[collection2].start + np.asarray(
                    [positions[collection2][int(column)] for column in collection2._idx[signal2]],
                    dtype=int,
                )
                cross = np.asarray(common_class.get_phicross(signal1, signal2, params))
                cross = np.diag(cross) if cross.ndim == 1 else cross
                phi_common[np.ix_(rows, columns)] += cross
                phi_common[np.ix_(columns, rows)] += cross.T

        return phi_common

    def map_params(self, xs):
        xs = np.asarray(xs)
        expected = sum(p.size if p.size is not None else 1 for p in self.params)
        if xs.ndim != 1 or len(xs) != expected:
            raise ValueError(f"expected a flat vector of {expected} PTA parameters, " f"received shape {xs.shape}")
        ret = {}
        ct = 0
        for p in self.params:
            if p.size is None:
                ret[p.name] = float(xs[ct])
                ct += 1
            else:
                ret[p.name] = np.asarray(xs[ct : ct + p.size])
                ct += p.size
        return ret

    def get_lnprior(self, params):
        # map parameter vector if needed
        params = params if isinstance(params, dict) else self.map_params(params)

        return np.sum([p.get_logpdf(params=params) for p in self.params])

    @property
    def pulsars(self):
        return [p.psrname for p in self._signalcollections]

    def get_hypercube_transform(self, params):
        # transform from unit cube to prior cube for nested sampling using PPFs
        # map parameter vector if needed
        params = params if isinstance(params, dict) else self.map_params(params)

        return np.hstack([p.get_ppf(params=params) for p in self.params])

    def _set_signal_dict(self):
        """Set signal dictionary"""

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
        """Return signal dictionary."""
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
                if "BasisGP" in sig.__class__.__name__:
                    summary += "\nBasis shape (Ntoas x N basis functions): {}".format(str(sig.get_basis().shape))
                    summary += "\nN selected toas: {}\n".format(str(len([i for i in sig._masks[0] if i])))
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
            """Sets the cache for various signal types."""

            self.white_params = []
            self.basis_params = []
            self.delay_params = []
            self.prior_params = []
            for signal in self._signals:
                if signal.signal_type == "white noise":
                    self.white_params.extend(signal.ndiag_params)
                elif signal.signal_type in ["basis", "common basis"]:
                    # to support GP coefficients, and yet do the right thing
                    # for common GPs, which do not have coefficients yet
                    self.delay_params.extend(getattr(signal, "delay_params", []))
                    self.basis_params.extend(signal.basis_params)
                    self.prior_params.extend(getattr(signal, "prior_params", []))
                elif signal.signal_type in ["deterministic"]:
                    self.delay_params.extend(signal.delay_params)
                else:
                    msg = "{} signal type not recognized! Caching ".format(signal.signal_type)
                    msg += "may not work correctly for this signal."
                    logger.error(msg)

        # def cache_clear(self):
        #     for instance in [self] + self.signals:
        #         kill = [attr for attr in instance.__dict__ if attr.startswith("_cache")]
        #
        #        for attr in kill:
        #            del instance.__dict__[attr]

        # a candidate for memoization
        @property
        def params(self):
            return sorted({param for signal in self._signals for param in signal.params}, key=lambda par: par.name)

        @property
        def param_names(self):
            ret = []
            for p in self.params:
                if p.size is not None:
                    for ii in range(0, p.size):
                        ret.append(p.name + "_{}".format(ii))
                else:
                    ret.append(p.name)
            return ret

        @property
        def signals(self):
            return self._signals

        def __repr__(self):
            return "<Enterprise SignalCollection object " + self.psrname + ": " + ", ".join(self.keys()) + ">"

        # emulate a dictionary

        def __len__(self):
            return len(self._signals)

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._signals[key]
            else:
                for s in self._signals:
                    if s.signal_id == key:
                        return s

                raise KeyError("Signal model not found")

        def keys(self):
            return [s.signal_id for s in self._signals]

        def values(self):
            return self._signals

        def items(self):
            return [(s.signal_id, s) for s in self._signals]

        # set default parameters

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
                return getattr(self, par)
            else:
                raise AttributeError("{} object has no attribute {}".format(self.__class__, par))

        def _compute_basis_split(self, common_signals):
            """Split the basis into columns of common signals and all other columns."""

            common, seen = [], set()
            for signal in self._signals:
                if signal in common_signals and signal in self._idx:
                    for column in self._idx[signal]:
                        column = int(column)
                        if column not in seen:
                            seen.add(column)
                            common.append(column)

            all_columns = {
                int(column) for signal in self._signals if signal in self._idx for column in self._idx[signal]
            }
            pulsar = sorted(all_columns.difference(seen))
            return np.asarray(common, dtype=int), np.asarray(pulsar, dtype=int)

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
            if self._Fmat is None:
                return None

            Fmat = np.zeros_like(self._Fmat)

            for signal in self._signals:
                if signal in self._idx:
                    Fmat[:, self._idx[signal]] = signal.get_basis(params)

            return Fmat

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

        def get_basis_common(self, params, common_idx):
            """Return basis columns associated with common signals."""

            basis = self.get_basis(params)
            return basis[:, common_idx] if basis is not None and common_idx.size else None

        def get_basis_pulsar_only(self, params, pulsar_idx):
            """Return basis columns not associated with common signals."""

            basis = self.get_basis(params)
            return basis[:, pulsar_idx] if basis is not None and pulsar_idx.size else None

        def get_phi_common(self, params, common_idx):
            """Return the covariance for basis columns of common signals."""

            phi = self.get_phi(params)
            if phi is None or not common_idx.size:
                return None
            if phi.ndim == 1:
                return KernelMatrix(np.array(phi[common_idx]))
            return KernelMatrix(np.array(phi[np.ix_(common_idx, common_idx)]))

        def get_phiinv_pulsar_only(self, params, pulsar_idx, logdet=False):
            """Return the inverse covariance for columns not associated with common signals."""

            phi = self.get_phi(params)
            if phi is None or not pulsar_idx.size:
                return (None, 0.0) if logdet else None
            if phi.ndim == 1:
                phi = KernelMatrix(np.array(phi[pulsar_idx]))
            else:
                phi = KernelMatrix(np.array(phi[np.ix_(pulsar_idx, pulsar_idx)]))
            return phi.inv(logdet)

        @cache_call(["basis_params", "white_params", "delay_params"])
        def get_TNr(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=T)

        @cache_call(["basis_params", "white_params"])
        def get_TNT(self, params):
            T = self.get_basis(params)
            if T is None:
                return None
            Nvec = self.get_ndiag(params)
            return Nvec.solve(T, left_array=T)

        @cache_call(["white_params", "delay_params"])
        def get_rNr_logdet(self, params):
            Nvec = self.get_ndiag(params)
            res = self.get_detres(params)
            return Nvec.solve(res, left_array=res, logdet=True)

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

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        super(csc_matrix_alt, self).__init__(arg1, shape=shape, dtype=dtype, copy=copy)
        self._has_sqrtsolve = False

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

    def sqrtsolve(self, other, left_array=None):

        raise NotImplementedError("csc_matrix_alt does not implement sqrtsolve")


class ndarray_alt(np.ndarray):
    """Sub-class of ``np.ndarray`` with custom ``solve`` method."""

    def __new__(cls, inputarr):
        if inputarr.ndim != 1:
            raise NotImplementedError("ndarray_alt does not support non-diagonal arrays")

        obj = np.asarray(inputarr).view(cls)
        obj._has_sqrtsolve = True

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

    def sqrtsolve(self, other, left_array=None):
        if other.ndim == 1:
            mult = np.array(other / np.sqrt(self))
        elif other.ndim == 2:
            mult = np.array(other / np.sqrt(self[:, None]))
        if left_array is not None:
            mult = np.dot(left_array.T, mult)

        return mult


class BlockMatrix(object):
    def __init__(self, blocks, slices, nvec=0):
        self._blocks = blocks
        self._slices = slices
        self._idxs = [indices_from_slice(slc) for slc in slices]
        self._nvec = nvec
        self._has_sqrtsolve = False

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
        for idx, block in zip(self._idxs, self._blocks):
            Zblock = Z[idx, :]
            Xblock = X[idx, :]

            if len(idx) > 1:
                cf = sl.cho_factor(block + np.diag(self._nvec[idx]))
                bx = sl.cho_solve(cf, Xblock)
            else:
                bx = Xblock / self._nvec[idx][:, None]
            ZNX += np.dot(Zblock.T, bx)
        ZNX += ZNXr
        return ZNX.squeeze() if len(ZNX) > 1 else ZNX.astype(float)

    def _solve_NX(self, X):
        """Solves :math:`N^{-1}X`, where :math:`X`
        is a 1-d or 2-d array.
        """
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)

        NX = X / self._nvec[:, None]
        for idx, block in zip(self._idxs, self._blocks):
            Xblock = X[idx, :]
            if len(idx) > 1:
                cf = sl.cho_factor(block + np.diag(self._nvec[idx]))
                NX[idx] = sl.cho_solve(cf, Xblock)
        return NX.squeeze()

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        if len(self._idx) > 0:
            logdet = np.sum(np.log(self._nvec[self._idx]))
        else:
            logdet = 0
        for idx, block in zip(self._idxs, self._blocks):
            if len(idx) > 1:
                cf = sl.cho_factor(block + np.diag(self._nvec[idx]))
                logdet += np.sum(2 * np.log(np.diag(cf[0])))
            else:
                logdet += np.sum(np.log(self._nvec[idx]))
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

    def sqrtsolve(self, other, left_array=None):

        raise NotImplementedError("BlockMatrix does not implement sqrtsolve")


class ShermanMorrison(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        self._jvec = jvec
        self._slices = slices
        self._idxs = [indices_from_slice(slc) for slc in slices]
        self._nvec = nvec
        self._has_sqrtsolve = True

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
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                rblock = x[idx]
                niblock = 1 / self._nvec[idx]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                Nx[idx] -= beta * np.dot(niblock, rblock) * niblock
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                xblock = x[idx]
                yblock = y[idx]
                niblock = 1 / self._nvec[idx]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
        return yNx

    def _sqrtsolve_D2(self, x):
        """Apply :math:`N^{-1/2}x` where :math:`x` is a 2-d array.

        This uses the closed-form inverse-square-root for diagonal-plus-rank1
        ECORR blocks, rather than a Cholesky factor solve.
        """

        Lix = x / np.sqrt(self._nvec[:, None])
        for idx, jv in zip(self._idxs, self._jvec):
            d = self._nvec[idx]
            inv_d = 1.0 / d
            inv_sqrt_d = 1.0 / np.sqrt(d)

            v = jv * np.sum(inv_d)
            if v > 0.0:
                t = np.sqrt(1.0 + v)
                # Stable equivalent of (1/sqrt(1+v) - 1) / v
                alpha = -1.0 / (t * (t + 1.0))
            else:
                alpha = -0.5

            vtAmb = jv * np.einsum("i,ij->j", inv_d, x[idx, :])
            scale = alpha * vtAmb
            Lix[idx, :] += inv_sqrt_d[:, None] * scale[None, :]

        return Lix

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                Zblock = Z[idx, :]
                Xblock = X[idx, :]
                niblock = 1 / self._nvec[idx]
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
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                niblock = 1 / self._nvec[idx]
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
                raise NotImplementedError("ShermanMorrison does not implement _solve_D2")
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret

    def sqrtsolve(self, other, left_array=None):
        if other.ndim == 1:
            shape = other.shape
            ret = self._sqrtsolve_D2(other.reshape(-1, 1)).reshape(*shape)

            if left_array is not None and left_array.ndim == 1:
                ret = np.sum(left_array * ret)
            elif left_array is not None:
                raise NotImplementedError("ShermanMorrison does not implement _sqrtsolve_2D1")

        elif other.ndim == 2:
            if left_array is None:
                ret = self._sqrtsolve_D2(other)
            elif left_array is not None and left_array.ndim == 2:
                raise NotImplementedError("ShermanMorrison does not implement _sqrtsolve_2D2")
            elif left_array is not None and left_array.ndim == 1:
                raise NotImplementedError("ShermanMorrison does not implement _sqrtsolve_1D2")
            else:
                raise TypeError
        else:
            raise TypeError

        return ret
