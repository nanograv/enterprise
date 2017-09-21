# parameter.py
"""Contains parameter types for use in `enterprise` ``Signal`` classes."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import math, inspect, functools

import numpy as np
import scipy.stats

from enterprise.signals.selections import selection_func


class Parameter(object):
    # instances will need to define _size, _prior, and _typename
    # thus this class is technically abstract

    def __init__(self, name):
        self.name = name
        self.prior = self._prior(name)

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

    def sample(self, **kwargs):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    @property
    def params(self):
        return [self] + [par for par in self.prior.params
                         if not isinstance(par, ConstantParameter)]

    def __repr__(self):
        typename = self._typename.format(**self.prior._params)
        array = '' if self._size is None else '[{}]'.format(self._size)

        return '"{}":{}{}'.format(self.name, typename, array)

    # this trick lets us pass an instantiated parameter to a signal;
    # the parameter will refuse to be renamed and will return itself
    def __call__(self, name):
        return self


def UserParameter(prior, size=None):
    """Class factor for UserParameter, with prior given as an Enterprise
    function (one argument, the value; arbitrary keyword arguments, which become
    hyperparameters)."""

    class UserParameter(Parameter):
        _size = size
        _prior = prior
        _typename = 'UserParameter'

    return UserParameter


def _argrepr(typename, **kwargs):
    args = []
    for par, arg in kwargs.items():
        if type(arg) == type and issubclass(arg,Parameter):
            args.append('{}="{{{}.name}}"'.format(par, par))
        elif isinstance(arg,Parameter):
            args.append('{}={}'.format(par, arg.name))
        else:
            args.append('{}={}'.format(par, arg))

    return '{}({})'.format(typename,','.join(args))


def UniformPrior(value, pmin, pmax):
    """Prior function for LinearExp parameters."""

    if pmin >= pmax:
        raise ValueError("Uniform Parameter requires pmin < pmax.")

    if pmin <= value <= pmax:
        return 1.0 / (pmax - pmin)
    else:
        return 0.0

def Uniform(pmin, pmax, size=None):
    """Class factory for Uniform parameters."""

    class Uniform(Parameter):
        _size = size
        _prior = Function(UniformPrior, pmin=pmin, pmax=pmax)
        _typename = _argrepr('Uniform', pmin=pmin, pmax=pmax)

    return Uniform


# note: will not do a jointly normal prior
def NormalPrior(value, mu, sigma):
    """Prior function for Normal parameters."""

    return math.exp(-0.5 * (value - mu)**2 / sigma**2) / math.sqrt(2 * math.pi * sigma**2)

def Normal(mu=0, sigma=1, size=None):
    """Class factory for Normal parameters."""

    class Normal(Parameter):
        _size = size
        _prior = Function(NormalPrior, mu=mu, pmax=pmax)
        _typename = _argrepr('Normal', mu=mu, pmax=pmax)

    return Normal


def LinearExpPrior(value, pmin, pmax):
    """Prior function for LinearExp parameters."""

    if pmin >= pmax:
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    return np.log(10) * 10**value / (10**pmax - 10**pmin)

def LinearExp(pmin, pmax, size=None):
    """Class factory for LinearExp parameters."""

    class LinearExp(Parameter):
        _size = size
        _prior = Function(LinearExpPrior, pmin=pmin, pmax=pmax)
        _typename = _argrepr('LinearExp', pmin=pmin, pmax=pmax)

    return LinearExp


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

def Constant(val=None):
    class Constant(ConstantParameter):
        value = val

    return Constant


def Function(func, name='', **func_kwargs):
    fname = name

    class Function(object):
        def __init__(self, name, psr=None):
            self._func = selection_func(func)
            self._psr = psr

            self._params = {}
            self._defaults = {}

            # divide keyword parameters into those that are Parameter classes,
            # Parameter instances (useful for global parameters),
            # and something else (which we will assume is a value)
            for kw, arg in func_kwargs.items():
                if isinstance(arg, type) and issubclass(
                        arg, (Parameter, ConstantParameter)):
                    # parameter name template
                    # pname_[signalname_][fname_]parname
                    pnames = [name, fname, kw]
                    par = arg('_'.join([n for n in pnames if n]))
                    self._params[kw] = par
                elif isinstance(arg, (Parameter, ConstantParameter)):
                    self._params[kw] = arg
                else:
                    self._defaults[kw] = arg

        def __call__(self, *args, **kwargs):
            # order of parameter resolution:
            # - parameter given in kwargs
            # - named sampling parameter in self._params, if given in params
            #   or if it has a value
            # - parameter given as constant in Function definition
            # - default value for keyword parameter in func definition

            # trick to get positional arguments before params kwarg
            params = kwargs.get('params',{})
            if 'params' in kwargs:
                del kwargs['params']

            for kw, arg in func_kwargs.items():
                if kw not in kwargs and kw in self._params:
                    par = self._params[kw]

                    if par.name in params:
                        kwargs[kw] = params[par.name]
                    elif hasattr(par, 'value'):
                        kwargs[kw] = par.value

            for kw, arg in self._defaults.items():
                if kw not in kwargs:
                    kwargs[kw] = arg

            if self._psr is not None and 'psr' not in kwargs:
                kwargs['psr'] = self._psr
            return self._func(*args, **kwargs)

        def add_kwarg(self, **kwargs):
            self._defaults.update(kwargs)

        @property
        def params(self):
            # if we extract the ConstantParameter value above, we would not
            # need a special case here
            return sum([par.params for par in self._params.values()
                        if not isinstance(par, ConstantParameter)], [])

    return Function


def get_funcargs(func):
    """Convenience function to get args and kwargs of any function."""
    argspec = inspect.getargspec(func)
    if argspec.defaults is None:
        args = argspec.args
        kwargs = []
    else:
        args = argspec.args[:(len(argspec.args)-len(argspec.defaults))]
        kwargs = argspec.args[-len(argspec.defaults):]

    return args, kwargs


def function(func):
    """Decorator for Function."""

    funcargs, _ = get_funcargs(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fargs = {funcargs[ct]: val for ct, val in
                 enumerate(args[:len(funcargs)])}
        fargs.update(kwargs)
        if not np.all([fa in fargs.keys() for fa in funcargs]):
            return Function(func, **kwargs)
        for kw, arg in kwargs.items():
            if ((isinstance(arg, type) and issubclass(
                arg, (Parameter, ConstantParameter))) or isinstance(
                    arg, (Parameter, ConstantParameter))):
                return Function(func, **kwargs)
        return func(*args, **kwargs)

    return wrapper
