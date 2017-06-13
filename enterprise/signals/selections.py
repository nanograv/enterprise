# selections.py
"""Contains various selection functions to mask parameters by backend flags,
time-intervals, etc."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import inspect
import functools


def call_me_maybe(obj):
    """See `here`_ for description.

    .. _here: https://www.youtube.com/watch?v=fWNaR-rxAic
    """
    return obj() if hasattr(obj, '__call__') else obj


def selection_func(func):
    funcargs = inspect.getargspec(func).args

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        targs = list(args)

        # check for mask
        mask = kwargs.get('mask', Ellipsis)
        if 'mask' in kwargs:
            del kwargs['mask']

        if len(targs) < len(funcargs) and 'psr' in kwargs:
            psr = kwargs['psr']

            for funcarg in funcargs[len(args):]:
                if funcarg not in kwargs and hasattr(psr, funcarg):
                    targs.append(call_me_maybe(getattr(psr, funcarg))[mask])

        if 'psr' in kwargs and 'psr' not in funcargs:
            del kwargs['psr']

        return func(*targs, **kwargs)

    return wrapper


def Selection(func):
    """Class factory for TOA selection."""

    class Selection(object):
        def __init__(self, psr):
            self._psr = psr

        @property
        def masks(self):
            return selection_func(func)(psr=self._psr)

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
