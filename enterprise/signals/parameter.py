# parameter.py
"""Contains parameter types for use in `enterprise` ``Signal`` classes."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import scipy.stats
from enterprise.signals import prior


class Parameter(object):
    """Parameter base class."""

    def __init__(self, name):
        self.name = name

    def get_logpdf(self, value):
        return self._prior.logpdf(value)

    def get_pdf(self, value):
        return self._prior.pdf(value)

    def sample(self, size=1, random_state=None):
        return self._prior.sample(size, random_state)

    # this trick lets us pass an instantiated parameter to a signal;
    # the parameter will refuse to be renamed and will return itself
    def __call__(self, name):
        return self


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


def Uniform(pmin, pmax):
    """Class factory for Uniform parameters."""
    class Uniform(Parameter):
        _prior = prior.Prior(prior.UniformBoundedRV(pmin, pmax))

        def __repr__(self):
            return '"{}":Uniform({},{})'.format(self.name, pmin, pmax)

    return Uniform


def Normal(mu=0, sigma=1):
    """Class factory for Normal parameters."""
    class Normal(Parameter):
        _prior = prior.Prior(scipy.stats.norm(loc=mu, scale=sigma))

        def __repr__(self):
            return '"{}": Normal({},{})'.format(self.name, mu, sigma)

    return Normal


def Constant(val):
    class Constant(ConstantParameter):
        value = val
    return Constant
