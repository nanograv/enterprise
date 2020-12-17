# parameter.py
"""Contains parameter types for use in `enterprise` ``Signal`` classes."""

import functools
import inspect

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
            # sample hyperpars for this par, skip parameter itself
            parvalues.update(sample(par.params[1:]))

            parvalues[par.name] = par.sample(params=parvalues)


class Parameter(object):
    # instances will need to define _size, _prior (of _logprior, but not both),
    # and _typename thus this class is technically abstract

    def __init__(self, name):
        self.name = name

        if hasattr(self, "_prior"):
            self.prior = self._prior(name)
        elif hasattr(self, "_logprior"):
            self.logprior = self._logprior(name)
        else:
            msg = "Parameter classes need to define _prior, or _logprior."
            raise AttributeError(msg)

        self.type = self.__class__.__name__.lower()

    def get_logpdf(self, value=None, **kwargs):
        if not isinstance(self, Parameter):
            raise TypeError("You can only call get_logpdf() on an " "instantiated (named) Parameter.")

        if value is None and "params" in kwargs:
            value = kwargs["params"][self.name]

        if hasattr(self, "prior"):
            logpdf = np.log(self.prior(value, **kwargs))
        else:
            logpdf = self.logprior(value, **kwargs)

        return logpdf if self._size is None else np.sum(logpdf)

    def get_pdf(self, value=None, **kwargs):
        if not isinstance(self, Parameter):
            raise TypeError("You can only call get_pdf() on an " "instantiated (named) Parameter.")

        if value is None and "params" in kwargs:
            value = kwargs["params"][self.name]

        if hasattr(self, "prior"):
            pdf = self.prior(value, **kwargs)
        else:
            pdf = np.exp(self.logprior(value, **kwargs))

        return pdf if self._size is None else np.prod(pdf)

    def sample(self, **kwargs):
        if not isinstance(self, Parameter):
            raise TypeError("You can only call sample() on an " "instantiated (named) Parameter.")

        if self._sampler is None:
            raise AttributeError("No sampler was provided for this Parameter.")
        else:
            if self.name in kwargs:
                raise ValueError("You shouldn't give me my value when you're sampling me.!")

            if hasattr(self, "prior"):
                return self.prior(func=self._sampler, size=self._size, **kwargs)
            else:
                return self.logprior(func=self._sampler, size=self._size, **kwargs)

    @property
    def size(self):
        return self._size

    @property
    def params(self):
        priorparams = self.prior.params if hasattr(self, "prior") else self.logprior.params

        return [self] + [par for par in priorparams if not isinstance(par, ConstantParameter)]

    def __repr__(self):
        if hasattr(self, "prior"):
            args = self.prior._params.copy()
            args.update(self.prior._funcs)
        else:
            args = self.logprior._params.copy()
            args.update(self.logprior._funcs)

        typename = self._typename.format(**args)
        array = "" if self._size is None else "[{}]".format(self._size)

        return "{}:{}{}".format(self.name, typename, array)

    # this trick lets us pass an instantiated parameter to a signal;
    # the parameter will refuse to be renamed and will return itself
    def __call__(self, name):
        return self


def GPCoefficients(logprior, size):
    """Class factory for GP coefficients, which are usually created
    inside gp_signals.BasisGP."""

    class GPCoefficients(Parameter):
        _size = size
        _logprior = logprior
        _sampler = None  # MV: TO DO, connect with GP object
        _typename = "GPCoefficients"

    return GPCoefficients


def UserParameter(prior=None, logprior=None, sampler=None, size=None):
    """Class factory for UserParameter, implementing Enterprise parameters
    with arbitrary priors. The prior is specified by way of an Enterprise
    ``Function`` of the form ``prior(value, [par1, par2])``. Optionally,
    include ``sampler`` (a function with the same parameters as ``prior``),
    to allow random sampling of the parameter through
    ``enterprise.signals.parameter.sample``.

    :param prior:   parameter prior pdf, given as Enterprise ``Function``
    :param sampler: function returning a randomly sampled parameter according
                    to prior
    :param size:    length for vector parameter
    :return:        ``UserParameter`` class
    """

    class UserParameter(Parameter):
        _size = size
        if prior is not None:
            _prior = prior
        if logprior is not None:
            _logprior = logprior
        _sampler = None if sampler is None else staticmethod(sampler)
        _typename = "UserParameter"

    return UserParameter


def _argrepr(typename, **kwargs):
    args = []
    for par, arg in kwargs.items():
        if isinstance(arg, type) and issubclass(arg, (Parameter, FunctionBase)):
            args.append("{}={{{}}}".format(par, par))
        elif isinstance(arg, (Parameter, FunctionBase)):
            args.append("{}={}".format(par, arg))
        else:
            args.append("{}={}".format(par, arg))

    return "{}({})".format(typename, ", ".join(args))


def UniformPrior(value, pmin, pmax):
    """Prior function for Uniform parameters."""

    # we'll let scipy.stats handle errors in pmin/pmax specification
    # this handles vectors correctly, if pmin and pmax are scalars,
    # or if len(value) = len(pmin) = len(pmax)
    return scipy.stats.uniform.pdf(value, pmin, pmax - pmin)


def UniformSampler(pmin, pmax, size=None):
    """Sampling function for Uniform parameters."""

    # we'll let scipy.stats handle errors in pmin/pmax specification
    # this handles vectors correctly, if pmin and pmax are scalars,
    # or if len(value) = len(pmin) = len(pmax)
    return scipy.stats.uniform.rvs(pmin, pmax - pmin, size=size)


def Uniform(pmin, pmax, size=None):
    """Class factory for Uniform parameters (with pdf(x) ~ 1/[pmax - pmin]
    inside [pmin,pmax], 0 outside. Handles vectors correctly,
    if ``pmin`` and ``pmax`` are scalars,
    or if ``len(size) == len(pmin) == len(pmax)``

    :param pmin: minimum of uniform range
    :param pmax: maximum of uniform range
    :param size: length for vector parameter
    :return:     ``Uniform`` parameter class
    """

    class Uniform(Parameter):
        _size = size
        _prior = Function(UniformPrior, pmin=pmin, pmax=pmax)
        _sampler = staticmethod(UniformSampler)
        _typename = _argrepr("Uniform", pmin=pmin, pmax=pmax)

    return Uniform


def NormalPrior(value, mu, sigma):
    """Prior function for Normal parameters."""

    # we let scipy.stats handle parameter errors
    # this code handles vectors correctly, if mu and sigma are scalars,
    # if mu and sigma are vectors with len(value) = len(mu) = len(sigma),
    # or if len(value) = len(mu) and sigma is len(value) x len(value)
    cov = sigma if np.ndim(sigma) == 2 else sigma ** 2
    return scipy.stats.multivariate_normal.pdf(value, mean=mu, cov=cov)


def NormalSampler(mu, sigma, size=None):
    """Sampling function for Normal parameters."""

    if np.ndim(mu) == 1 and len(mu) != size:
        raise ValueError("Size mismatch between Parameter size and distribution arguments")

    # we let scipy.stats handle all other errors
    # this code handles vectors correctly, if mu and sigma are scalars,
    # if mu and sigma are vectors with len(value) = len(mu) = len(sigma),
    # or if len(value) = len(mu) and sigma is len(value) x len(value);
    # note that scipy.stats.multivariate_normal.rvs infers parameter
    # size from mu and sigma, so if these are vectors we pass size=None;
    # otherwise we'd get multiple copies of a jointly-normal vector
    cov = sigma if np.ndim(sigma) == 2 else sigma ** 2
    return scipy.stats.multivariate_normal.rvs(mean=mu, cov=cov, size=(None if np.ndim(mu) == 1 else size))


def Normal(mu=0, sigma=1, size=None):
    """Class factory for Normal parameters (with pdf(x) ~ N(``mu``,``sigma``)).
    Handles vectors correctly if ``size == len(mu) == len(sigma)``,
    in which case ``sigma`` is taken as the sqrt of the diagonal
    of the covariance matrix; ``sigma`` can also be given passed
    as the ``size`` x ``size`` covariance matrix.

    :param mu:    center of normal distribution
    :param sigma: standard deviation of normal distribution
    :param size:  length for vector parameter
    :return:      ``Normal`` parameter class
    """

    class Normal(Parameter):
        _size = size
        _prior = Function(NormalPrior, mu=mu, sigma=sigma)
        _sampler = staticmethod(NormalSampler)
        _typename = _argrepr("Normal", mu=mu, sigma=sigma)

    return Normal


def LinearExpPrior(value, pmin, pmax):
    """Prior function for LinearExp parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    # works with vectors if pmin and pmax are either scalars,
    # or len(value) vectors
    return ((pmin <= value) & (value <= pmax)) * np.log(10) * 10 ** value / (10 ** pmax - 10 ** pmin)


def LinearExpSampler(pmin, pmax, size):
    """Sampling function for LinearExp parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    # works with vectors if pmin and pmax are either scalars
    # or vectors, in which case one must have len(pmin) = len(pmax) = size
    return np.log10(np.random.uniform(10 ** pmin, 10 ** pmax, size))


def LinearExp(pmin, pmax, size=None):
    """Class factory for LinearExp parameters (with pdf(x) ~ 10^x,
    and 0 outside [``pmin``,``max``]). Handles vectors correctly
    if ``pmin`` and ``pmax`` are scalars or if
    ``size == len(pmin) == len(pmax)``

    :param pmin: minimum of range
    :param pmax: maximum of range
    :param size: length for vector parameter (default `None`)
    :return:     ``LinearExp`` parameter class
    """

    class LinearExp(Parameter):
        _size = size
        _prior = Function(LinearExpPrior, pmin=pmin, pmax=pmax)
        _sampler = staticmethod(LinearExpSampler)
        _typename = _argrepr("LinearExp", pmin=pmin, pmax=pmax)

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
        return "{}:Constant={}".format(self.name, self.value)


def Constant(val=None):
    """Class factory for Constant parameters.  Leave ``val=None`` to set
    value later, for example with ``signal_base.PTA.set_default_params()``.
    """

    class Constant(ConstantParameter):
        # MV: I don't know if this does what it's supposed to...
        value = val

    return Constant


class FunctionBase(object):
    pass


def Function(func, name="", **func_kwargs):
    fname = name

    class Function(FunctionBase):
        def __init__(self, name, psr=None):
            self._func = selection_func(func)
            self._psr = psr

            self._params = {}
            self._defaults = {}
            self._funcs = {}

            self.name = "_".join([n for n in [name, fname] if n])

            try:
                self.func_args = inspect.getfullargspec(func).args
            except:
                self.func_args = inspect.getargspec(func).args
            self.func_kwargs = func_kwargs

            # process keyword parameters:
            # - if they are Parameter classes, then we will instantiate
            #   them to named Parameter instances (using the Function name,
            #   if given, and the keyword), and save them to the
            #   self._params dictionary, using the keyword as key
            # - if they are Parameter instances, we will save them directly
            #   to self._params
            # - if they are Function classes, then we will instantiate
            #   them, save them to self._funcs, and add all of their
            #   parameters to self._params
            # - if they are something else, we will assume they are values,
            #   which we will save in self._defaults

            for kw, arg in self.func_kwargs.items():
                if isinstance(arg, type) and issubclass(arg, (Parameter, ConstantParameter)):

                    # parameter name template:
                    #   pname_[signalname_][fname_]parname
                    pnames = [name, fname, kw]
                    par = arg("_".join([n for n in pnames if n]))

                    self._params[kw] = par
                elif isinstance(arg, (Parameter, ConstantParameter)):
                    self._params[kw] = arg
                elif isinstance(arg, type) and issubclass(arg, FunctionBase):
                    # instantiate the function
                    pnames = [name, fname, kw]
                    parfunc = arg("_".join([n for n in pnames if n]), psr)

                    self._funcs[kw] = parfunc
                    self._params.update(parfunc._params)
                elif isinstance(arg, FunctionBase):
                    self._funcs[kw] = arg
                    self._params.update(arg._params)
                else:
                    self._defaults[kw] = arg

        def __call__(self, *args, **kwargs):
            # we call self._func (or possibly the `func` given in kwargs)
            # by passing it args, kwargs, after augmenting kwargs (see below)

            # kwargs['params'] is special, take it out of kwargs
            params = kwargs.get("params", {})
            if "params" in kwargs:
                del kwargs["params"]

            # if kwargs['func'] is given, we will call that instead
            func = kwargs.get("func", self._func)
            if "func" in kwargs:
                del kwargs["func"]

            # we augment kwargs as follows:
            # - parameters given in the original Function definition
            #   (and therefore included in func_kwargs), that are included
            #   in self._params, and given in kwargs['params']
            # - parameters given in the original Function definition
            #   (and therefore included in func_kwargs), that are included
            #   in self._params, and have a value attribute (e.g., Constants)
            # - parameters given as constants in the original Function
            #   definition (they are included in func_kwargs, and saved in
            #   self._defaults)
            # - parameters given as Functions, evaluated by passing
            #   them only the parameters they may care about
            # - [if the func itself has default parameters, they may yet
            #   apply if none of the above does]
            for kw, arg in self.func_kwargs.items():
                if kw not in kwargs:
                    if kw in self._params:
                        par = self._params[kw]

                        if par.name in params:
                            kwargs[kw] = params[par.name]
                        elif hasattr(par, "value"):
                            kwargs[kw] = par.value
                    elif kw in self._defaults:
                        kwargs[kw] = self._defaults[kw]
                    elif kw in self._funcs:
                        f = self._funcs[kw]
                        fargs = {par: val for par, val in kwargs.items() if par in f.func_kwargs}
                        fargs["params"] = params
                        kwargs[kw] = f(**fargs)

            # pass our pulsar if we have one
            if self._psr is not None and "psr" not in kwargs:
                kwargs["psr"] = self._psr

            # clean up parameters that are not meant for `func`
            # keep those required for `selection_func` to work
            # keep also `size` needed by samplers

            kwargs = {
                par: val
                for par, val in kwargs.items()
                if (par in self.func_kwargs or par in self.func_args or par in ["psr", "mask", "size"])
            }

            return func(*args, **kwargs)

        def add_kwarg(self, **kwargs):
            self._defaults.update(kwargs)

        @property
        def params(self):
            # if we extract the ConstantParameter value above, we would not
            # need a special case here
            return sum([par.params for par in self._params.values() if not isinstance(par, ConstantParameter)], [])

        def __repr__(self):
            return "{}({})".format(self.name, ", ".join([str(p) for p in self.params]))

    return Function


def get_funcargs(func):
    """Convenience function to get args and kwargs of any function."""
    try:
        argspec = inspect.getfullargspec(func)
    except:
        argspec = inspect.getargspec(func)

    if argspec.defaults is None:
        args = argspec.args
        kwargs = []
    else:
        args = argspec.args[: (len(argspec.args) - len(argspec.defaults))]
        kwargs = argspec.args[-len(argspec.defaults) :]

    return args, kwargs


def function(func):
    """Decorator for Function."""

    # get the positional arguments
    funcargs, _ = get_funcargs(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # make a dictionary of positional arguments (declared for func,
        # and passed to wrapper), and of keyword arguments passed to wrapper
        fargs = {funcargs[ct]: val for ct, val in enumerate(args[: len(funcargs)])}
        fargs.update(kwargs)

        # if any of the positional arguments are missing, we make a Function
        if not all(fa in fargs.keys() for fa in funcargs):
            return Function(func, **kwargs)

        # if any of the keyword arguments are Parameters or Functions,
        # we make a Function
        for kw, arg in kwargs.items():
            if (
                isinstance(arg, type)
                and issubclass(arg, (Parameter, ConstantParameter))
                or isinstance(arg, (Parameter, ConstantParameter))
                or isinstance(arg, type)
                and issubclass(arg, FunctionBase)
                or isinstance(arg, FunctionBase)
            ):

                return Function(func, **kwargs)

        # otherwise, we simply call the function
        return func(*args, **kwargs)

    return wrapper
