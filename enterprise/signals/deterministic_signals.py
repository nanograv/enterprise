# deterministic_signals.py
"""Contains class factories for deterministic signals.
Determinisitc signals are defined as the class of signals that have a
delay that is to be subtracted from the residuals.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

import enterprise.signals.signal_base as base
from enterprise.signals import selections
from enterprise.signals.selections import Selection


def Deterministic(waveform, selection=Selection(selections.no_selection),
                  name=''):
    """Class factory for generic deterministic signals."""

    class Deterministic(base.Signal):
        signal_type = 'deterministic'
        signal_name = name

        def __init__(self, psr):
            self._do_selection(psr, waveform, selection)

        def _do_selection(self, psr, waveform, selection):

            sel = selection(psr)
            self._keys = list(sorted(sel.masks.keys()))
            self._masks = [sel.masks[key] for key in self._keys]
            self._delay = np.zeros(len(psr.toas))
            self._wf, self._params = {}, {}
            for key, mask in zip(self._keys, self._masks):
                pnames = [psr.name, name, key]
                pname = '_'.join([n for n in pnames if n])
                self._wf[key] = waveform(pname, psr=psr)
                params = self._wf[key]._params.values()
                for param in params:
                    self._params[param.name] = param

        @property
        def delay_params(self):
            """Get any varying ndiag parameters."""
            return [pp.name for pp in self.params]

        @base.cache_call('delay_params')
        def get_delay(self, params):
            """Return signal delay."""
            for key, mask in zip(self._keys, self._masks):
                self._delay[mask] = self._wf[key](params=params, mask=mask)
            return self._delay

    return Deterministic
