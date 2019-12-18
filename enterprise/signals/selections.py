# selections.py
"""Contains various selection functions to mask parameters by backend flags,
time-intervals, etc."""

from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import inspect

import numpy as np


def call_me_maybe(obj):
    """See `here`_ for description.

    .. _here: https://www.youtube.com/watch?v=fWNaR-rxAic
    """
    return obj() if hasattr(obj, "__call__") else obj


def selection_func(func):
    try:
        funcargs = inspect.getfullargspec(func).args
    except:
        funcargs = inspect.getargspec(func).args

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        targs = list(args)

        # check for mask
        mask = kwargs.get("mask", Ellipsis)
        if "mask" in kwargs:
            del kwargs["mask"]

        if len(targs) < len(funcargs) and "psr" in kwargs:
            psr = kwargs["psr"]

            for funcarg in funcargs[len(args) :]:
                if funcarg not in kwargs and hasattr(psr, funcarg):
                    attr = call_me_maybe(getattr(psr, funcarg))
                    if isinstance(attr, np.ndarray) and getattr(mask, "shape", [0])[0] == len(attr):
                        targs.append(attr[mask])
                    else:
                        targs.append(attr)

        if "psr" in kwargs and "psr" not in funcargs:
            del kwargs["psr"]

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
            return {key: val * arr for key, val in masks.items()}

        def __call__(self, parname, parameter, arr=None):
            params, kmasks = {}, {}
            for key, val in self.masks.items():
                kname = "_".join([key, parname]) if key else parname
                pname = "_".join([self._psr.name, kname])
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
    """Selection function to split by data segment"""
    midpoint = (toas.max() + toas.min()) / 2
    return dict(zip(["t1", "t2"], [toas <= midpoint, toas > midpoint]))


def by_band(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    flagvals = np.unique(flags["B"])
    return {flagval: flags["B"] == flagval for flagval in flagvals}


def by_backend(backend_flags):
    """Selection function to split by backend flags."""
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def nanograv_backends(backend_flags):
    """Selection function to split by NANOGRav backend flags only."""
    flagvals = np.unique(backend_flags)
    ngb = ["ASP", "GASP", "GUPPI", "PUPPI"]
    flagvals = filter(lambda x: any(map(lambda y: y in x, ngb)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def custom_backends(cb):
    def backends(backend_flags):
        """Selection function to split by custom backend flags only.
        cb : list of str of the backends
        use None to recover by_backend
        use ["ASP", "GASP", "GUPPI", "PUPPI"] to recover nanograv_backends
        """
        flagvals = np.unique(backend_flags)
        if cb is not None:
            cb = list(np.atleast_1d(cb))
            flagvals = filter(lambda x: any(map(lambda y: y in x, cb)),
                              flagvals)
        else:
            pass
        return {flagval: backend_flags == flagval for flagval in flagvals}

    return backends


def no_selection(toas):
    """Default selection with no splitting."""
    return {"": np.ones_like(toas, dtype=bool)}
