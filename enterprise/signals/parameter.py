# parameter.py
"""Contains parameter types for use in `enterprise` ``Signal`` classes."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

import scipy.stats
from enterprise.signals import prior


class Parameter(object):
    """Parameter base class."""

    _size = None

    def __init__(self, name):
        self.name = name

    def get_logpdf(self, value=None, params={}):
        if value is None:
            value = params[self.name]

        logpdf = self._prior.logpdf(value)
        return logpdf if self._size is None else np.sum(logpdf)

    def get_pdf(self, value=None, params={}):
        if value is None:
            value = params[self.name]

        pdf = self._prior.pdf(value)
        return pdf if self._size is None else np.prod(pdf)

    def sample(self, n=1, random_state=None):
        if self._size is None:
            s = self._prior.sample(n, random_state)
            if n == 1:
                s = float(s)
        else:
            if random_state is None:
                if n > 1:
                    s = self._prior.sample(n * self._size).reshape(
                        (n,self._size))
                else:
                    s = self._prior.sample(self._size)
            else:
                raise NotImplementedError(
                    "Currently cannot handle random_state when size > 1")

        return s

    @property
    def size(self):
        return self._size

    @property
    def params(self):
        return [self]

    # this trick lets us pass an instantiated parameter to a signal;
    # the parameter will refuse to be renamed and will return itself
    def __call__(self, name):
        return self


def UserParameter(prior, size=None):
    """Class factor for UserParameter, with prior given as an Enterprise
    function (one argument, arbitrary keyword arguments, which become
    hyperparameters."""

    class UserParameter(Parameter):
        _size = size

        def __init__(self, name):
            super(UserParameter, self).__init__(name)
            self.prior = prior(name)

        def get_logpdf(self, value=None, **kwargs):
            if value is None and 'params' in kwargs:
                value = kwargs['params'][self.name]
                del kwargs['params'][self.name]

            logpdf = np.log(self.prior(value, **kwargs))
            return logpdf if self._size is None else np.sum(logpdf)

        def get_pdf(self, value=None, **kwargs):
            if value is None and 'params' in kwargs:
                value = kwargs['params'][self.name]
                del kwargs['params'][self.name]

            pdf = self.prior(value, **kwargs)
            return pdf if self._size is None else np.prod(pdf)

        def sample(self, *args):
            raise NoteImplementedError(
                "Currently cannot sample UserParameters")

        @property
        def params(self):
            return [self] + [par for par in self.prior.params
                             if not isinstance(par, ConstantParameter)]

        def __repr__(self):
            return '"{}":UserParameter({})'.format(self.name,
                ','.join(param.name for param in self.params[1:]))

    return UserParameter


class ConstantParameter(object):
    """Constant Parameter base class."""

    def __init__(self, name):
        self.name = name

    @property
    def value(self):
        return self.value

    @value.setter
    def value(self, value):
        self.value = value

    def __call__(self, name):
        return self

    def __repr__(self):
        return '"{}":Constant={}'.format(self.name, self.value)


def Uniform(pmin, pmax, size=None):
    """Class factory for Uniform parameters."""
    class Uniform(Parameter):
        _prior = prior.Prior(prior.UniformBoundedRV(pmin, pmax))
        _size = size

        def __repr__(self):
            return '"{}":Uniform({},{})'.format(self.name, pmin, pmax) \
                + ('' if self._size is None else '[{}]'.format(self._size))

    return Uniform


def LinearExp(pmin, pmax, size=None):
    """Class factory for LinearExp parameters."""
    class LinearExp(Parameter):
        _prior = prior.Prior(prior.LinearExpRV(pmin, pmax))
        _size = size

        def __repr__(self):
            return '"{}":LinearExp({},{})'.format(self.name, pmin, pmax) \
                + ('' if self._size is None else '[{}]'.format(self._size))

    return LinearExp


def Normal(mu=0, sigma=1, size=None):
    """Class factory for Normal parameters."""
    class Normal(Parameter):
        _prior = prior.Prior(scipy.stats.norm(loc=mu, scale=sigma))
        _size = size

        def __repr__(self):
            return '"{}": Normal({},{})'.format(self.name, mu, sigma) \
                + ('' if self._size is None else '[{}]'.format(self._size))

    return Normal


def Constant(val=None):
    class Constant(ConstantParameter):
        value = val

    return Constant
