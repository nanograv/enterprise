# signals.py

"""
Defines the basic signal model interace classes. We define three base
classes at the moment to model stochastic (i.e. red noise, ECORR, etc),
white noise (i.e. EFAC, EQUAD, etc) and deterministic signals
(i.e. timing model, wavelets, etc).

Since these are base classes any derived signal classes must implement
signal-specific functions.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np


class StochasticSignal(object):
    """Base class for stochastic signals."""

    def __init__(self):

        self._param_dict = {}
        self.signal_type = 'stochastic'

    def setup(self, psr, *args, **kwargs):
        """Abstract class for setting up signal model auxiliaries.

        Must be created by sub-classes.

        :param psr: Instance of ``Pulsar`` class.
        """
        pass

    def add_param(self, param):
        """Add parameter to signal model.

        :param param: Instance of ``Parameter`` class.
        """
        self._param_dict[param.name] = param

    def update_basis_functions(self, params):
        """Update basis functions with new parameters if necessary.

        Must be created by sub-classes.

        :param params: Parameter vector.

        .. note: May want to use a different way of inputting parameters.
        """
        pass

    def get_kernel_inv_det(self, params):
        """Compute the inverse of the Gaussian kernel.

        This method computes the log-determinant and cholesky factor
        of the rank reduced component of the gaussian kernel matrix
        :math:`K = TBT^T`, where :math:`B` is the rank-reduced kernel
        matrix. We compute :math:`\mathrm{logdet}(B)` and :math:`B^{-1}`.

        This method must be implemented by sub-classes

        :param params: Parameter vector.

        .. note: May want to use a different way of inputting parameters.

        """
        pass

    @property
    def param_vector(self):
        """Get vector of parameters in signal model."""
        return np.array([p.value for p in self._param_dict.values()])

    @property
    def params(self):
        """Return list of parameter names."""
        return self._param_dict.keys()
