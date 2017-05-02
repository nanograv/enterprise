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


def EcorrBasisModel(log10_ecorr=parameter.Uniform(-10, -5), by_backend=False):
    class EcorrBasisModel(base.Signal):
        signal_type = 'basis'
        signal_name = 'ecorr'

        def __init__(self, psr):

            avetoas, aveflags, self._F = util.create_quantization_matrix(
                psr.toas, psr.backend_flags, dt=1)

            if by_backend:
                self._params, self._jvec = util.get_masked_data(
                    psr.name, 'log10_ecorr', log10_ecorr, aveflags,
                    np.ones_like(avetoas))
            else:
                self._params = {'log10_ecorr':
                                log10_ecorr(psr.name + '_log10_ecorr')}
                self._jvec = {'log10_ecorr':np.ones_like(avetoas)}

        def get_basis(self, params=None):
            return self._F

        def get_phi(self, params):
            ret = np.sum([10**(2*self.get(p, params))*self._jvec[p]
                          for p in self._params], axis=0)
            return ret

        def get_phiinv(self, params):
            return 1 / self.get_phi(params)

        @property
        def basis_shape(self):
            return self._F.shape

    return EcorrBasisModel
