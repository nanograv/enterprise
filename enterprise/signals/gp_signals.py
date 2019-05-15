# gp_signals.py
"""Contains class factories for Gaussian Process (GP) signals.
GP signals are defined as the class of signals that have a basis
function matrix and basis prior vector..
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import math
import itertools
import functools

import numpy as np

from enterprise.signals import signal_base
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import utils
from enterprise.signals.parameter import function
from enterprise.signals.selections import Selection
from enterprise.signals.utils import KernelMatrix


def BasisGP(priorFunction, basisFunction, coefficients=False, combine=True,
            selection=Selection(selections.no_selection),
            name=''):
    """Class factory for generic GPs with a basis matrix."""

    class BasisGP(signal_base.Signal):
        signal_type = 'basis'
        signal_name = name
        signal_id = name

        basis_combine = combine

        def __init__(self, psr):
            super(BasisGP, self).__init__(psr)
            self.name = self.psrname + '_' + self.signal_id
            self._do_selection(psr, priorFunction, basisFunction,
                               coefficients, selection)

        def _do_selection(self, psr, priorfn, basisfn, coefficients,
                          selection):
            sel = selection(psr)

            self._keys = list(sorted(sel.masks.keys()))
            self._masks = [sel.masks[key] for key in self._keys]
            self._prior, self._bases = {}, {}
            self._params, self._coefficients = {}, {}

            for key, mask in zip(self._keys, self._masks):
                pnames = [psr.name, name, key]
                pname = '_'.join([n for n in pnames if n])

                self._prior[key] = priorfn(pname, psr=psr)
                self._bases[key] = basisfn(pname, psr=psr)

                for par in itertools.chain(self._prior[key]._params.values(),
                                           self._bases[key]._params.values()):
                    self._params[par.name] = par

            if coefficients:
                # we can only create GPCoefficients parameters if the basis
                # can be constructed with default arguments
                # (and does not change size)
                self._construct_basis()

                for key in self._keys:
                    pname = '_'.join([n for n in [psr.name, name, key] if n])

                    chain = itertools.chain(self._prior[key]._params.values(),
                                            self._bases[key]._params.values())
                    priorargs = {par.name: self._params[par.name]
                                 for par in chain}

                    logprior = parameter.Function(
                        functools.partial(self._get_coefficient_logprior, key),
                        **priorargs)

                    size = self._slices[key].stop - self._slices[key].start

                    cpar = parameter.GPCoefficients(
                        logprior=logprior, size=size)(pname + '_coefficients')

                    self._coefficients[key] = cpar
                    self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            ret = []
            for basis in self._bases.values():
                ret.extend([pp.name for pp in basis.params])
            return ret

        @signal_base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            basis, self._labels = {}, {}
            for key, mask in zip(self._keys, self._masks):
                basis[key], self._labels[key] = self._bases[key](
                    params=params, mask=mask)

            nc = np.sum(F.shape[1] for F in basis.values())
            self._basis = np.zeros((len(self._masks[0]), nc))
            self._phi = KernelMatrix(nc)
            self._slices = {}
            nctot = 0
            for key, mask in zip(self._keys, self._masks):
                Fmat = basis[key]
                nn = Fmat.shape[1]
                self._basis[mask, nctot:nn+nctot] = Fmat
                self._slices.update({key: slice(nctot, nn+nctot)})
                nctot += nn

        # this class does different things (and gets different method
        # definitions) if the user wants it to model GP coefficients
        # (e.g., for a hierarchical likelihood) or if they do not
        if coefficients:
            def _get_coefficient_logprior(self, key, c, **params):
                self._construct_basis(params)

                phi = self._prior[key](self._labels[key],params=params)

                if phi.ndim == 1:
                    return (-0.5 * np.sum(c * c / phi) -
                            0.5 * np.sum(np.log(phi)) -
                            0.5 * len(phi) * np.log(2*math.pi))
                    # note: (2*pi)^(n/2) is not in signal_base likelihood
                else:
                    # TO DO: this code could be embedded in KernelMatrix
                    phiinv, logdet = KernelMatrix(phi).inv(logdet=True)
                    return (-0.5 * np.dot(c,np.dot(phiinv,c)) -
                            0.5 * logdet -
                            0.5 * phi.shape[0] * np.log(2*math.pi))

            # MV: could assign this to a data member at initialization
            @property
            def delay_params(self):
                return [pp.name for pp in self.params
                        if '_coefficients' in pp.name]

            @signal_base.cache_call(['basis_params', 'delay_params'])
            def get_delay(self, params={}):
                self._construct_basis(params)

                c = np.zeros(self._basis.shape[1])
                for key, slc in self._slices.items():
                    p = self._coefficients[key]
                    c[slc] = params[p.name] if p.name in params else p.value

                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phiinv(self, params):
                return None
        else:
            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)

                for key, slc in self._slices.items():
                    phislc = self._prior[key](
                        self._labels[key], params=params)
                    self._phi = self._phi.set(phislc, slc)
                return self._phi

            def get_phiinv(self, params):
                return self.get_phi(params).inv()

    return BasisGP


def FourierBasisGP(spectrum, coefficients=False, combine=True, components=20,
                   selection=Selection(selections.no_selection),
                   Tspan=None, modes=None, name='red_noise'):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_red(nmodes=components,
                                                Tspan=Tspan, modes=modes)
    BaseClass = BasisGP(spectrum, basis, coefficients=coefficients,
                        combine=combine, selection=selection, name=name)

    class FourierBasisGP(BaseClass):
        signal_type = 'basis'
        signal_name = 'red noise'
        signal_id = name

    return FourierBasisGP


def TimingModel(coefficients=False, name='linear_timing_model',
                use_svd=False, normed=True):
    """Class factory for marginalized linear timing model signals."""

    if normed is True:
        basis = utils.normed_tm_basis()
    elif isinstance(normed, np.ndarray):
        basis = utils.normed_tm_basis(norm=normed)
    elif use_svd is True:
        if normed is not True:
            msg = "use_svd == True is incompatible with normed != True"
            raise ValueError(msg)
        basis = utils.svd_tm_basis()
    else:
        basis = utils.unnormed_tm_basis()

    prior = utils.tm_prior()
    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = 'basis'
        signal_name = 'linear timing model'
        signal_id = name + '_svd' if use_svd else name

        if coefficients:
            def _get_coefficient_logprior(self, key, c, **params):
                # MV: probably better to avoid this altogether
                #     than to use 1e40 as in get_phi
                return 0

    return TimingModel


@function
def ecorr_basis_prior(weights, log10_ecorr=-8):
    """Returns the ecorr prior.
    :param weights: A vector or weights for the ecorr prior.
    """
    return weights * 10**(2*log10_ecorr)


def EcorrBasisModel(log10_ecorr=parameter.Uniform(-10, -5),
                    coefficients=False,
                    selection=Selection(selections.no_selection),
                    name='basis_ecorr'):
    """Convenience function to return a BasisGP class with a
    quantized ECORR basis."""

    basis = utils.create_quantization_matrix()
    prior = ecorr_basis_prior(log10_ecorr=log10_ecorr)
    BaseClass = BasisGP(prior, basis, coefficients=coefficients,
                        selection=selection, name=name)

    class EcorrBasisModel(BaseClass):
        signal_type = 'basis'
        signal_name = 'basis ecorr'
        signal_id = name

    return EcorrBasisModel


def BasisCommonGP(priorFunction, basisFunction, orfFunction,
                  coefficients=False, combine=True, name=''):

    class BasisCommonGP(signal_base.CommonSignal):
        signal_type = 'common basis'
        signal_name = 'common'
        signal_id = name

        basis_combine = combine

        _orf = orfFunction(name)
        _prior = priorFunction(name)

        def __init__(self, psr):
            super(BasisCommonGP, self).__init__(psr)
            self.name = self.psrname + '_' + self.signal_id

            pname = '_'.join([psr.name, name])
            self._bases = basisFunction(pname, psr=psr)

            self._params, self._coefficients = {}, {}

            for par in itertools.chain(self._prior._params.values(),
                                       self._orf._params.values(),
                                       self._bases._params.values()):
                self._params[par.name] = par

            self._psrpos = psr.pos

            if coefficients:
                self._construct_basis()

                chain = itertools.chain(self._prior._params.values(),
                                        self._orf._params.values(),
                                        self._bases._params.values())
                priorargs = {par.name: self._params[par.name]
                             for par in chain}

                logprior = parameter.Function(
                    self._get_coefficient_logprior,
                    **priorargs)

                size = self._basis.shape[1]

                cpar = parameter.GPCoefficients(
                    logprior=logprior, size=size)(pname + '_coefficients')

                self._coefficients[''] = cpar
                self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            return [pp.name for pp in self._bases.params]

        @signal_base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            self._basis, self._labels = self._bases(params=params)

        if coefficients:
            def _get_coefficient_logprior(self, c, **params):
                # MV: for correlated GPs, the prior needs to use
                #     the coefficients for all GPs together;
                #     this may require parameter groups

                raise NotImplementedError("Need to implement common prior " +
                                          "for BasisCommonGP coefficients")

            @property
            def delay_params(self):
                return [pp.name for pp in self.params
                        if '_coefficients' in pp.name]

            @signal_base.cache_call(['basis_params', 'delay_params'])
            def get_delay(self, params={}):
                self._construct_basis(params)

                p = self._coefficients['']
                c = params[p.name] if p.name in params else p.value
                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phicross(cls, signal1, signal2, params):
                return None

            def get_phiinv(self, params):
                return None
        else:
            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)

                prior = BasisCommonGP._prior(self._labels, params=params)
                orf = BasisCommonGP._orf(self._psrpos, self._psrpos,
                                         params=params)

                return prior * orf

            @classmethod
            def get_phicross(cls, signal1, signal2, params):
                prior = BasisCommonGP._prior(signal1._labels, params=params)
                orf = BasisCommonGP._orf(signal1._psrpos, signal2._psrpos,
                                         params=params)

                return prior * orf

    return BasisCommonGP


def FourierBasisCommonGP(spectrum, orf, coefficients=False, combine=True,
                         components=20, Tspan=None, modes=None, name='common_fourier'):

    if coefficients and Tspan is None:
        raise ValueError("With coefficients=True, FourierBasisCommonGP " +
                         "requires that you specify Tspan explicitly.")

    basis = utils.createfourierdesignmatrix_red(nmodes=components,
                                                Tspan=Tspan, modes=modes)
    BaseClass = BasisCommonGP(spectrum, basis, orf,
                              coefficients=coefficients, combine=combine,
                              name=name)

    class FourierBasisCommonGP(BaseClass):
        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        @signal_base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            span = (Tspan if Tspan is not None else
                    max(FourierBasisCommonGP._Tmax) -
                    min(FourierBasisCommonGP._Tmin))
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP


# for simplicity, we currently do not handle Tspan automatically
def FourierBasisCommonGP_ephem(spectrum, components, Tspan, name='ephem_gp'):
    basis = utils.createfourierdesignmatrix_ephem(nmodes=components,
                                                  Tspan=Tspan)
    orf = utils.monopole_orf()

    return BasisCommonGP(spectrum, basis, orf, name=name)


def FourierBasisCommonGP_physicalephem(frame_drift_rate=1e-9,
                                       d_jupiter_mass=1.54976690e-11,
                                       d_saturn_mass=8.17306184e-12,
                                       d_uranus_mass=5.71923361e-11,
                                       d_neptune_mass=7.96103855e-11,
                                       jup_orb_elements=0.05,
                                       sat_orb_elements=0.5,
                                       coefficients=False,
                                       name='physicalephem_gp'):
    """
    Class factory for physical ephemeris corrections as a common GP.
    Individual perturbations can be excluded by setting the corresponding
    prior sigma to None.

    :param frame_drift_rate: Gaussian sigma for frame drift rate
    :param d_jupiter_mass:   Gaussian sigma for Jupiter mass perturbation
    :param d_saturn_mass:    Gaussian sigma for Saturn mass perturbation
    :param d_uranus_mass:    Gaussian sigma for Uranus mass perturbation
    :param d_neptune_mass:   Gaussian sigma for Neptune mass perturbation
    :param jup_orb_elements: Gaussian sigma for Jupiter orbital elem. perturb.
    :param sat_orb_elements: Gaussian sigma for Saturn orbital elem. perturb.
    :param coefficients:     if True, treat GP coefficients as enterprise
                             parameters; if False, marginalize over them

    :return: BasisCommonGP representing ephemeris perturbations
    """

    basis = utils.createfourierdesignmatrix_physicalephem(
        frame_drift_rate=frame_drift_rate,
        d_jupiter_mass=d_jupiter_mass,
        d_saturn_mass=d_saturn_mass,
        d_uranus_mass=d_uranus_mass,
        d_neptune_mass=d_neptune_mass,
        jup_orb_elements=jup_orb_elements,
        sat_orb_elements=sat_orb_elements)

    spectrum = utils.physicalephem_spectrum()
    orf = utils.monopole_orf()

    return BasisCommonGP(spectrum, basis, orf,
                         coefficients=coefficients, name=name)
