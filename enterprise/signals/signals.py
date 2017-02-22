# signals.py

# Defines the basic signal model interace classes.

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from enterprise.signals.parameter import Parameter


class StochasticSignal(object):
    """Base class for stochastic signals."""

    def __init__(self):

        self._param_dict = {}
        self.signal_type = None

    def setup(self, psr):
        """Abstract class for setting up signal model auxiliaries.

        Must be created by sub-classes.

        :param psr: Instance of `Pulsar` class.
        """
        pass

    def add_param(self, param):
        """Add parameter to signal model.

        :param param: Instance of `Parameter` class.
        """
        self._param_dict[param.name] = param

    @property
    def param_vector(self):
        """Get vector of parameters in signal model."""
        return np.array([p.value for p in self._param_dict.values()])

    @property
    def params(self):
        """Returns list of parameter names."""
        return self._param_dict.keys()

