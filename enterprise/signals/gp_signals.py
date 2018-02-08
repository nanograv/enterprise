# gp_signals.py
"""Contains class factories for Gaussian Process (GP) signals.
GP signals are defined as the class of signals that have a basis
function matrix and basis prior vector..
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from enterprise.signals import utils
from enterprise.signals import parameter
import enterprise.signals.signal_base as base
from enterprise.signals import selections
from enterprise.signals.selections import Selection


def BasisGP(priorFunction, basisFunction,
            selection=Selection(selections.no_selection),
            name=''):
    """Class factory for generic GPs with a basis matrix."""

    class BasisGP(base.Signal):
        signal_type = 'basis'
        signal_name = name

        def __init__(self, psr):
            super(BasisGP, self).__init__(psr)
            self.name = self.psrname + '_' + self.signal_name
            self._do_selection(psr, priorFunction, basisFunction, selection)

        def _do_selection(self, psr, priorfn, basisfn, selection):

            sel = selection(psr)
            self._keys = list(sorted(sel.masks.keys()))
            self._masks = [sel.masks[key] for key in self._keys]
            self._prior, self._bases, self._params = {}, {}, {}
            for key, mask in zip(self._keys, self._masks):
                pnames = [psr.name, name, key]
                pname = '_'.join([n for n in pnames if n])
                self._prior[key] = priorfn(pname, psr=psr)
                self._bases[key] = basisfn(pname, psr=psr)
                params = sum([list(self._prior[key]._params.values()),
                              list(self._bases[key]._params.values())],[])
                for param in params:
                    self._params[param.name] = param

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            ret = []
            for basis in self._bases.values():
                ret.extend([pp.name for pp in basis.params])
            return ret

        @base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            basis, self._labels = {}, {}
            for key, mask in zip(self._keys, self._masks):
                basis[key], self._labels[key] = self._bases[key](
                    params=params, mask=mask)

            nc = np.sum(F.shape[1] for F in basis.values())
            self._basis = np.zeros((len(self._masks[0]), nc))
            self._phi = base.KernelMatrix(nc)
            self._slices = {}
            nctot = 0
            for key, mask in zip(self._keys, self._masks):
                Fmat = basis[key]
                nn = Fmat.shape[1]
                self._basis[mask, nctot:nn+nctot] = Fmat
                self._slices.update({key: slice(nctot, nn+nctot)})
                nctot += nn

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


def FourierBasisGP(spectrum, components=20,
                   selection=Selection(selections.no_selection),
                   Tspan=None, name=''):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan)
    BaseClass = BasisGP(spectrum, basis, selection=selection, name=name)

    class FourierBasisGP(BaseClass):
        signal_type = 'basis'
        signal_name = 'red_noise_' + name if name else 'red_noise'

    return FourierBasisGP


def TimingModel(name='linear_timing_model'):
    """Class factory for marginalized linear timing model signals."""

    class TimingModel(base.Signal):
        signal_type = 'basis'
        signal_name = name

        def __init__(self, psr):
            super(TimingModel, self).__init__(psr)
            self.name = self.psrname + '_' + name
            self._params = {}

            self._F = psr.Mmat.copy()

            norm = np.sqrt(np.sum(self._F**2, axis=0))
            self._F /= norm

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params=None):
            return np.ones(self._F.shape[1])*1e40

        def get_phiinv(self, params=None):
            return 1 / self.get_phi(params)

        @property
        def basis_shape(self):
            return self._F.shape

        #TODO: this is somewhat of a hack until we get this class more general
        @property
        def basis_params(self):
            return []

    return TimingModel


@base.function
def ecorr_basis_prior(weights, log10_ecorr=-8):
    """Returns the ecorr prior.
    :param weights: A vector or weights for the ecorr prior.
    """
    return weights * 10**(2*log10_ecorr)


def EcorrBasisModel(log10_ecorr=parameter.Uniform(-10, -5),
                    selection=Selection(selections.no_selection),
                    name=''):
    """Convienience function to return a BasisGP class with a
    quantized ECORR basis."""

    basis = utils.create_quantization_matrix()
    prior = ecorr_basis_prior(log10_ecorr=log10_ecorr)
    BaseClass = BasisGP(prior, basis, selection=selection, name=name)

    class EcorrBasisModel(BaseClass):
        signal_type = 'basis'
        signal_name = 'basis_ecorr_' + name if name else 'basis_ecorr'

    return EcorrBasisModel


def BasisCommonGP(priorFunction, basisFunction, orfFunction, name=''):

    class BasisCommonGP(base.CommonSignal):
        signal_type = 'common basis'
        signal_name = name
        _orf = orfFunction(name)
        _prior = priorFunction(name)

        def __init__(self, psr):
            super(BasisCommonGP, self).__init__(psr)
            self.name = self.psrname + '_' + self.signal_name

            self._bases = basisFunction(psr.name+name, psr=psr)
            params = sum([list(BasisCommonGP._prior._params.values()),
                          list(BasisCommonGP._orf._params.values()),
                          list(self._bases._params.values())], [])
            self._params = {}
            for param in params:
                self._params[param.name] = param

            self._psrpos = psr.pos

        @base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            self._basis, self._labels = self._bases(params=params)

        def get_basis(self, params={}):
            self._construct_basis(params)
            return self._basis

        def get_phi(self, params):
            self._construct_basis(params)
            prior = BasisCommonGP._prior(
                self._labels, params=params)
            orf = BasisCommonGP._orf(self._psrpos, self._psrpos, params=params)
            return prior * orf

        @classmethod
        def get_phicross(cls, signal1, signal2, params):
            prior = BasisCommonGP._prior(signal1._labels,
                                         params=params)
            orf = BasisCommonGP._orf(signal1._psrpos, signal2._psrpos,
                                     params=params)
            return prior * orf

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            return [pp.name for pp in self._bases.params]

    return BasisCommonGP


def FourierBasisCommonGP(spectrum, orf, components=20,
                         Tspan=None, name=''):

    basis = utils.createfourierdesignmatrix_red(nmodes=components)
    BaseClass = BasisCommonGP(spectrum, basis, orf, name=name)

    class FourierBasisCommonGP(BaseClass):
        signal_name = 'common_fourier_' + name if name else 'common_fourier'

        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        @base.cache_call('basis_params')
        def _construct_basis(self, params={}):
            span = (Tspan if Tspan is not None else
                    max(FourierBasisCommonGP._Tmax) -
                    min(FourierBasisCommonGP._Tmin))
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP
