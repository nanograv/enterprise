# selections.py
"""Contains various selection functions to mask parameters by backend flags,
time-intervals, etc."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import inspect
import functools

from enterprise.pulsar import Pulsar


def call_me_maybe(obj):
    """See `here`_ for description.

    .. _here: https://www.youtube.com/watch?v=fWNaR-rxAic
    """
    return obj() if hasattr(obj, '__call__') else obj


def selection_func(func):
    funcargs = inspect.getargspec(func).args

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0],Pulsar):
            # extract function arguments from Pulsar object
            # (calling methods as needed)
            # but allow overriding them with keyword arguments

            pulsarargs = {funcarg: call_me_maybe(getattr(args[0],funcarg))
                          for funcarg in funcargs if hasattr(args[0],funcarg)}
            pulsarargs.update(**kwargs)

            return func(**pulsarargs)
        else:
            return func(*args,**kwargs)

    return wrapper


def Selection(func):
    """Class factory for TOA selection."""

    class Selection(object):
        def __init__(self, psr):
            self._psr = psr

        @property
        def masks(self):
            return selection_func(func)(self._psr)

        def _get_masked_array_dict(self, masks, arr):
            return {key: val*arr for key, val in masks.items()}

        def __call__(self, parname, parameter, arr=None):
            params, kmasks = {}, {}
            for key, val in self.masks.items():
                kname = '_'.join([parname, key]) if key else parname
                pname = '_'.join([self._psr.name, kname])
                params.update({kname: parameter(pname)})
                kmasks.update({kname: val})

            if arr is not None:
                ma = self._get_masked_array_dict(kmasks, arr)
                ret = (params, ma)
            else:
                ret = params, kmasks
            return ret

    return Selection


# SELECTION FUNCTIONS

def cut_half(toas):
    midpoint = (toas.max() + toas.min()) / 2
    return dict(zip(['t1', 't2'], [toas <= midpoint, toas > midpoint]))


def by_backend(backend_flags):
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def no_selection(toas):
    return {'': np.ones_like(toas, dtype=bool)}
