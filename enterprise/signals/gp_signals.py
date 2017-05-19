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


def FourierBasisGP(spectrum, components=20,
                   selection=Selection(selections.no_selection), Tspan=None):
    """Class factory for fourier basis GPs."""

    class FourierBasisGP(base.Signal):
        signal_type = 'basis'
        signal_name = 'red noise'

        def __init__(self, psr):

            # TODO: this could be cleaned up...
            sel = selection(psr)
            masks = sel.masks
            self._spectrum = {}
            self._f2 = {}
            self._params = {}
            Fmats = {}
            for key in sorted(masks.keys()):
                mask = masks[key]
                self._spectrum[key] = spectrum(psr.name, key)
                Fmats[key], self._f2[key], _ = \
                    utils.createfourierdesignmatrix_red(
                        psr.toas[mask], components, freq=True, Tspan=Tspan)
                for param in self._spectrum[key]._params.values():
                    self._params[param.name] = param

            nf = np.sum(F.shape[1] for F in Fmats.values())
            self._F = np.zeros((len(psr.toas), nf))
            self._phi = np.zeros(nf)
            self._slices = {}
            nftot = 0
            for key in sorted(masks.keys()):
                mask = masks[key]
                Fmat = Fmats[key]
                nn = Fmat.shape[1]
                self._F[mask, nftot:nn+nftot] = Fmat
                self._slices.update({key: slice(nftot, nn+nftot)})
                nftot += nn

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            for key, slc in self._slices.items():
                self._phi[slc] = self._spectrum[key](
                    self._f2[key], **params) * self._f2[key][0]
            return self._phi

        def get_phiinv(self, params):
            return 1 / self.get_phi(params)

        @property
        def basis_shape(self):
            return self._F.shape

    return FourierBasisGP


def TimingModel():
    """Class factory for marginalized linear timing model signals."""

    class TimingModel(base.Signal):
        signal_type = 'basis'
        signal_name = 'linear timing model'

        def __init__(self, psr):
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

    return TimingModel


def EcorrBasisModel(log10_ecorr=parameter.Uniform(-10, -5),
                    selection=Selection(selections.no_selection)):

    class EcorrBasisModel(base.Signal):
        signal_type = 'basis'
        signal_name = 'ecorr'

        def __init__(self, psr):

            # TODO: this could be cleaned up and probably done in one loop...
            sel = selection(psr)
            self._params, self._masks = sel('log10_ecorr', log10_ecorr)
            Umats = {}
            for key in sorted(self._masks.keys()):
                mask = self._masks[key]
                Umats.update({key: utils.create_quantization_matrix(
                    psr.toas[mask])})
            nepoch = np.sum(U.shape[1] for U in Umats.values())
            self._F = np.zeros((len(psr.toas), nepoch))
            netot = 0
            self._jvec = {}
            self._phi = np.zeros(nepoch)
            for key in sorted(self._masks.keys()):
                mask = self._masks[key]
                Umat = Umats[key]
                nn = Umat.shape[1]
                self._F[mask, netot:nn+netot] = Umat
                self._jvec.update({key: slice(netot, nn+netot)})
                netot += nn

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            for p in self._params:
                self._phi[self._jvec[p]] = 10**(2*self.get(p, params))
            return self._phi

        def get_phiinv(self, params):
            return 1 / self.get_phi(params)

        @property
        def basis_shape(self):
            return self._F.shape

    return EcorrBasisModel


def FourierBasisCommonGP(crossspectrum=None, components=20,
                         Tspan=None, name='common'):

    class FourierBasisCommonGP(base.CommonSignal):
        signal_type = 'basis'
        signal_name = 'red noise'

        _crossspectrum = crossspectrum(name)
        _Tmin, _Tmax = [], []

        def __init__(self, psr):

            self._params = FourierBasisCommonGP._crossspectrum._params
            self._psrpos = psr.pos
            self._toas = psr.toas

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        # goofy way to cache the basis, there may be a better way?
        def __getattr__(self, par):
            if par in ['_f2', '_F']:
                span = (Tspan if Tspan else max(FourierBasisCommonGP._Tmax) -
                        min(FourierBasisCommonGP._Tmin))
                self._F, self._f2, _ = utils.createfourierdesignmatrix_red(
                    self._toas, components, freq=True, Tspan=span)

                return getattr(self, par)
            else:
                raise AttributeError('{} object has no attribute {}'.format(
                    self.__class__,par))

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            # note multiplying by f[0] is not general
            return FourierBasisCommonGP._crossspectrum(
                self._f2, self._psrpos, self._psrpos, **params) * self._f2[0]

        @classmethod
        def get_phicross(cls, signal1, signal2, params):
            # currently pass the pulsar objects, what else could we do?
            # note multiplying by f[0] is not general
            return FourierBasisCommonGP._crossspectrum(
                signal1._f2, signal1._psrpos, signal2._psrpos,
                **params) * signal1._f2[0]

    return FourierBasisCommonGP
