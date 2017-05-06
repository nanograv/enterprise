# gp_signals.py
"""Contains class factories for Gaussian Process (GP) signals.
GP signals are defined as the class of signals that have a basis
function matrix and basis prior vector..
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

import enterprise.signals.utils as util
from enterprise.signals import parameter
import enterprise.signals.signal_base as base
from enterprise.signals import selections
from enterprise.signals.selections import Selection


def FourierBasisGP(spectrum, components=20):
    """Class factory for fourier basis GPs."""

    class FourierBasisGP(base.Signal):
        signal_type = 'basis'
        signal_name = 'red noise'

        def __init__(self, psr):
            self._spectrum = spectrum(psr.name)
            self._params = self._spectrum._params

            self._toas = psr.toas
            self._T = np.max(self._toas) - np.min(self._toas)

            self._F, self._f2, _ = util.createfourierdesignmatrix_red(
                self._toas, nmodes=components, freq=True)

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            return self._spectrum(self._f2, **params) / self._T

        def get_phiinv(self, params):
            return self._T / self._spectrum(self._f2, **params)

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

            sel = selection(psr, 'log10_ecorr', log10_ecorr)
            self._params, self._masks = sel()
            Umats = []
            for key, mask in self._masks.items():
                Umats.append(util.create_quantization_matrix(psr.toas[mask]))
            nepoch = np.sum(U.shape[1] for U in Umats)
            self._F = np.zeros((len(psr.toas), nepoch))
            netot = 0
            self._jvec = {}
            for (key, mask), Umat in zip(self._masks.items(), Umats):
                nn = Umat.shape[1]
                self._F[mask, netot:nn+netot] = Umat
                self._jvec.update({key:np.ones(nn)})
                netot += nn

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            ret = np.hstack([10**(2*self.get(p, params))*self._jvec[p]
                             for p in self._params])
            return ret

        def get_phiinv(self, params):
            return 1 / self.get_phi(params)

        @property
        def basis_shape(self):
            return self._F.shape

    return EcorrBasisModel
