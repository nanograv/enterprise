# white_signals.py
"""Contains class factories for white noise signals. White noise signals are
defined as the class of signals that only modifies the white noise matrix `N`.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

import enterprise.signals.utils as util
from enterprise.signals import parameter
import enterprise.signals.signal_base as base


def MeasurementNoise(efac=parameter.Uniform(0.5,1.5), by_backend=False):
    """Class factory for EFAC type measurement noise."""

    class MeasurementNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'efac'

        def __init__(self, psr):

            if by_backend:
                self._params, self._ndiag = util.get_masked_data(
                    psr.name, 'efac', efac, psr.backend_flags, psr.toaerrs**2)
            else:
                self._params = {'efac': efac(psr.name + '_efac')}
                self._ndiag = {'efac':psr.toaerrs**2}

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [self.get(p, params)**2*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return MeasurementNoise


def EquadNoise(log10_equad=parameter.Uniform(-10,-5), by_backend=False):
    """Class factory for EQUAD type measurement noise."""

    class EquadNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'equad'

        def __init__(self,psr):

            if by_backend:
                self._params, self._ndiag = util.get_masked_data(
                    psr.name, 'log10_equad', log10_equad, psr.backend_flags,
                    np.ones_like(psr.toaerrs))
            else:
                self._params = {'log10_equad':
                                log10_equad(psr.name + '_log10_equad')}
                self._ndiag = {'log10_equad':np.ones_like(psr.toaerrs)}

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [10**(2*self.get(p, params))*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return EquadNoise
