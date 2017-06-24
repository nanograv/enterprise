# prior.py
"""
Defines classes used for evaluation of prior probabilities.
Currently this is only capable of handling independent 1D priors on
each parameter.  We may want to include joint ND prior distributions
for collections of correlated parameters...
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import scipy.stats
from scipy.stats import rv_continuous
#TODO consider 'rv_discrete' for empirical prior

import numpy as np


class Prior(object):
    r"""A class for Priors.
    A Prior object can return the probability density using the ``pdf()`` and
    ``logpdf()`` methods.  These may take scalars or numpy arrays as
    arguments.

    Note that Prior instances contain only the ``pdf()`` and ``logpdf()``
    methods. They do not retain all of the original ``rv_continuous``
    methods and attributes.

    """

    def __init__(self, rv):
        """
        :param rv: rv_frozen Private member that holds an instance of
                      rv_frozen used to define the prior distribution.
                      It must be a 'frozen distribution', with all shape
                      parameters (i.e. a, b, loc, scale) set.

        For more info see `rv_continuous`_.
        For a list of functions suitable for use see
        `continuous-distributions`_.

        :Examples:

        .. code-block:: python

            # A half-bounded uniform prior on Agw, ensuring Agw >= 0.
            model.Agw.prior = Prior(UniformUnnormedRV(lower=0.))

            # A normalized uniform prior on Agw in [10^-18, 10^-12]
            model.Agw.prior = Prior(UniformBoundedRV(1.0e-18, 1.0e-12))

            # A Gaussian prior for gamma = 13/3 +/- 2/3
            mean, std = 13./3., 2./3.
            model.gamma.prior = Prior(scipy.stats.norm(loc=mean, scale=std))

            # A bounded Gaussian prior to ensure that eccentrity is in [0, 1]
            mean, std, low, up = 0.9, 0.1, 0.0, 1.0
            model.ecc.prior = Prior(GaussinaBoundedRV(loc=mean, scale=std,
                                                      lower=low, upper=up))

        .. _rv_continuous: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
        .. _continuous-distributions: https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        .. # noqa E501
        """
        self._rv = rv
        pass

    def pdf(self, value):
        """Return the probability distribution function at a specified value.

        :param value: Input value of parameter.
        :type value: float
        :return: pdf at input value.

        """
        # The astype() calls prevent unsafe cast messages
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.pdf(v)

    def logpdf(self, value):
        """Return the logarithm of probability distribution function
        at a specified value.

        :param value: Input value of parameter.
        :type value: float
        :return: log-pdf at input value.

        """
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.logpdf(v)

    def sample(self, size=1, random_state=None):
        """Return a sample from probability distribution function.
        This calls the `rvs` method of the underlying
        `rv_continuous` class.

        :param size: Defining number of random variates (default is 1).
        :type size: int or tuple of ints
        :random_state: If int or RandomState, use it for drawing the random
                    variates. If None, rely on `self.random_state`.
                    Default is None.
        :type random_state: None or int or `np.random.RandomState` instance

        :return: random variate of size
        :rtype: ndarray or float
        """
        return self._rv.rvs(size=size, random_state=random_state)


class _linear_exp_gen(rv_continuous):
    """A distribution with pdf(x) ~ 10^x."""

    def _rvs(self):
        return np.log10(np.random.uniform(10**self.a, 10**self.b, self._size))

    def _pdf(self, x):
        return np.log(10) * 10**x / (10**self.b-10**self.a)

    def _logpdf(self, x):
        return np.log(self._pdf(x))

    def _cdf(self, x):
        return np.log(10) * (10**self.a - 10**x) / (10**self.a - 10**self.b)


class _UniformUnnormedRV_generator(rv_continuous):
    r"""An unnormalized, uniform prior distribution set to unity
    everywhere.  This should be used for unbounded or half-bounded
    intervals.
    """
    # The astype() calls prevent unsafe cast messages
    def _pdf(self, x):
        return np.ones_like(x).astype(np.float64, casting='same_kind')

    def _logpdf(self, x):
        return np.zeros_like(x).astype(np.float64, casting='same_kind')

    def _rvs(self):
        raise RuntimeError('cannot sample from unnormed distribution')


def LinearExpRV(lower=0, upper=1):
    r""" A prior proportional to 10^x with lower and upper bounds."""
    return _linear_exp_gen(a=lower, b=upper)


def UniformUnnormedRV(lower=-np.inf, upper=np.inf):
    r"""An unnormalized uniform prior suitable for unbounded or
    half-bounded intervals.

    :param lower: lower bound of parameter range
    :type lower: float
    :param upper: upper bound of parameter range
    :type upper: float
    :return: a frozen rv_continuous instance with pdf equal to unity
               in the allowed interval and 0 elsewhere
    """
    return _UniformUnnormedRV_generator(name='unnormed_uniform',
                                        a=lower, b=upper)


def UniformBoundedRV(lower=0., upper=1.):
    r"""A uniform prior between two finite bounds.
    This is a convenience function with more natural bound parameters
    than ``scipy.stats.uniform``.

    :param lower: lower bound of parameter range
    :type lower: float
    :param upper: upper bound of parameter range
    :type upper: float
    :return: a frozen rv_continuous instance with normalized uniform
               probability inside the range [lower, upper] and 0 outside
    """
    uu = scipy.stats.uniform(lower, (upper - lower))
    return uu


def GaussianBoundedRV(loc=0., scale=1., lower=0., upper=1.):
    r"""A Gaussian prior between two finite bounds.
    This is a convenience function with more natural bound parameters
    than ``scipy.stats.truncnorm``.

    :param loc: location parameter, mean of distribution
    :type loc: float
    :param scale: standard deviation of distribution
    :type scale: float
    :param lower: lower bound of parameter range
    :type lower: float
    :param upper: upper bound of parameter range
    :type upper: float
    :return: a frozen rv_continuous instance with normalized Gaussian
               probability truncated to the range [lower, upper] and 0 outside
    """
    low, up = (lower - loc) / scale, (upper - loc) / scale
    nn = scipy.stats.truncnorm(loc=loc, scale=scale, a=low, b=up)
    return nn
