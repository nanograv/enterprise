# parameter.py
"""Contains parameter types for use in `enterprise` ``Signal`` classes."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import inspect
import functools

import numpy as np
import scipy.stats

from enterprise.signals.selections import selection_func


def sample(parlist):
    """Sample a list of Parameters consistently (i.e., keeping
    track of hyperparameters)."""

    # we'll be nice and accept a single parameter
    parlist = [parlist] if isinstance(parlist, Parameter) else parlist

    ret = {}
    _sample(parlist, ret)

    return ret


def _sample(parlist, parvalues):
    """Recursive function used by sample()."""

    for par in parlist:
        if par not in parvalues:
            parvalues.update(sample(par.params[1:]))
            parvalues[par.name] = par.sample(params=parvalues)


class Parameter(object):
    # instances will need to define _size, _prior, and _typename
    # thus this class is technically abstract

    def __init__(self, name):
        self.name = name
        self.prior = self._prior(name)

    def get_logpdf(self, value=None, **kwargs):
        if value is None and 'params' in kwargs:
            value = kwargs['params'][self.name]

        logpdf = np.log(self.prior(value, **kwargs))
        return logpdf if self._size is None else np.sum(logpdf)

    def get_pdf(self, value=None, **kwargs):
        if value is None and 'params' in kwargs:
            value = kwargs['params'][self.name]

        pdf = self.prior(value, **kwargs)
        return pdf if self._size is None else np.prod(pdf)

    def sample(self, **kwargs):
        if self._sampler is None:
            raise AttributeError("No sampler was provided for this Parameter.")
        else:
            if self.name in kwargs:
                raise ValueError(
                    "You shouldn't give me my value when you're sampling me.!")

            return self.prior(func=self._sampler, size=self._size, **kwargs)

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


def UserParameter(prior, sampler=None, size=None):
    """Class factory for UserParameter, with `prior` given as an Enterprise
    Function (one argument, the value; arbitrary keyword arguments, which
    become hyperparameters). Optionally, `sampler` can be given as a regular
    (not Enterprise function), taking the same keyword parameters as `prior`.
    """

    class UserParameter(Parameter):
        _size = size
        _prior = prior
        _sampler = None if sampler is None else staticmethod(sampler)
        _typename = 'UserParameter'

    return UserParameter


def _argrepr(typename, **kwargs):
    args = []
    for par, arg in kwargs.items():
        if type(arg) == type and issubclass(arg, Parameter):
            args.append('{}="{{{}.name}}"'.format(par, par))
        elif isinstance(arg, Parameter):
            args.append('{}={}'.format(par, arg.name))
        else:
            args.append('{}={}'.format(par, arg))

    return '{}({})'.format(typename,','.join(args))


def UniformPrior(value, pmin, pmax):
    """Prior function for Uniform parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("Uniform Parameter requires pmin < pmax.")

    return scipy.stats.uniform.pdf(value, pmin, (pmax - pmin))


def UniformSampler(pmin, pmax, size=None):
    """Sampling function for Uniform parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("Uniform Parameter requires pmin < pmax.")

    return scipy.stats.uniform.rvs(pmin, pmax-pmin, size=size)


def Uniform(pmin, pmax, size=None):
    """Class factory for Uniform parameters."""

    class Uniform(Parameter):
        _size = size
        _prior = Function(UniformPrior, pmin=pmin, pmax=pmax)
        _sampler = staticmethod(UniformSampler)
        _typename = _argrepr('Uniform', pmin=pmin, pmax=pmax)

    return Uniform


# note: will not do a jointly normal prior
def NormalPrior(value, mu, sigma):
    """Prior function for Normal parameters. Note that `sigma` can be a
    scalar for a 1-d distribution, a vector for multivariate distribution that
    uses the vector as the sqrt of the diagonal of the covaraince matrix,
    or a matrix which is the covariance."""

    cov = sigma if np.ndim(sigma) == 2 else sigma**2
    return scipy.stats.multivariate_normal.pdf(value, mean=mu, cov=cov)


def NormalSampler(mu, sigma, size=None):
    """Sampling function for Normal parameters."""

    cov = sigma if np.ndim(sigma) == 2 else sigma**2
    return scipy.stats.multivariate_normal.rvs(
        mean=mu, cov=cov, size=size)


def Normal(mu=0, sigma=1, size=None):
    """Class factory for Normal parameters."""

    class Normal(Parameter):
        _size = size
        _prior = Function(NormalPrior, mu=mu, sigma=sigma)
        _sampler = staticmethod(NormalSampler)
        _typename = _argrepr('Normal', mu=mu, sigma=sigma)

    return Normal


def LinearExpPrior(value, pmin, pmax):
    """Prior function for LinearExp parameters."""

    if pmin >= pmax:
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    return (((pmin <= value) & (value <= pmax)) * np.log(10) *
            10**value / (10**pmax - 10**pmin))


def LinearExpSampler(pmin, pmax, size):
    """Sampling function for LinearExp parameters."""

    if pmin >= pmax:
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    return np.log10(np.random.uniform(10**pmin, 10**pmax, size))


def LinearExp(pmin, pmax, size=None):
    """Class factory for LinearExp parameters (with pdf(x) ~ 10^x)."""

    class LinearExp(Parameter):
        _size = size
        _prior = Function(LinearExpPrior, pmin=pmin, pmax=pmax)
        _sampler = staticmethod(LinearExpSampler)
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


class Functional(object):
    pass


def Function(func, name='', **func_kwargs):
    fname = name

    class Function(Functional):
        def __init__(self, name, psr=None):
            self._func = selection_func(func)
            self._psr = psr

            self._params = {}
            self._defaults = {}
            # self._funcs = {}

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
                # elif isinstance(arg, type) and issubclass(
                #         arg, Functional):
                #     pnames = [name, fname, kw]
                #     parfunc = arg('_'.join([n for n in pnames if n]), psr)
                #     self._funcs[kw] = parfunc
                #     self._params.update(parfunc._params)
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
            params = kwargs.get('params', {})
            if 'params' in kwargs:
                del kwargs['params']

            # allow calling an alternate function with the same parameters
            func = kwargs.get('func', self._func)
            if 'func' in kwargs:
                del kwargs['func']

            # for kw, arg in func_kwargs.items():
            #     if kw not in kwargs and kw in self._funcs:
            #         # note: should we look for psr first?
            #         kwargs[kw] = self._funcs[kw](params=kwargs)

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

            return func(*args, **kwargs)

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
