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
                   selection=Selection(selections.no_selection)):
    """Class factory for fourier basis GPs."""

    class FourierBasisGP(base.Signal):
        signal_type = 'basis'
        signal_name = 'red noise'

        def __init__(self, psr):

            # TODO: this could be cleaned up...
            sel = selection(psr)
            self._spectrum = {}
            self._f2 = {}
            self._params = {}
            Fmats = {}
            for key, mask in sel.masks.items():
                self._spectrum[key] = spectrum(psr.name, key)
                Fmats[key], self._f2[key], _ = \
                    utils.createfourierdesignmatrix_red(
                        psr.toas[mask], components, freq=True)
                for param in self._spectrum[key]._params.values():
                    self._params[param.name] = param

            nf = np.sum(F.shape[1] for F in Fmats.values())
            self._F = np.zeros((len(psr.toas), nf))
            self._phi = np.zeros(nf)
            self._slices = {}
            nftot = 0
            for key, mask in sel.masks.items():
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

            self._F = psr.Mmat

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
            for key, mask in self._masks.items():
                Umats.update({key: utils.create_quantization_matrix(
                    psr.toas[mask])})
            nepoch = np.sum(U.shape[1] for U in Umats.values())
            self._F = np.zeros((len(psr.toas), nepoch))
            netot = 0
            self._jvec = {}
            self._phi = np.zeros(nepoch)
            for key, mask in self._masks.items():
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
