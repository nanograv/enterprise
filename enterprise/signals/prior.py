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
    A Prior object can return the probability density using the `pdf()` and
    `logpdf()` methods.  These may take scalars or numpy arrays as arguements.

    Note that Prior instances contain only the `pdf()` and `logpdf()` methods
    They do not retain all of the original `rv_continuous` methods and
    attributes.
    """

    def __init__(self, rv):
        """
        : param _rv : rv_frozen Private member that holds an instance of
                      rv_frozen used to define the prior distribution.
                      It must be a 'frozen distribution', with all shape
                      parameters (i.e. a, b, loc, scale) set.

        For more info see <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>  # noqa: E501
        For a list of functions suitable for use see <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>  # noqa: E501

        Examples
        --------
        # A half-bounded uniform prior on Agw, ensuring Agw >= 0.
        model.Agw.prior = Prior(UniformUnnormedRV(lower=0.))

        # A normalized uniform prior on Agw in [10^-18, 10^-12]
        model.Agw.prior = Prior(UniformBoundedRV(1.0e-18, 1.0e-12))

        # A bounded Gaussian prior to ensure that eccentrity is in [0, 1]
        mean, std, lower, upper = 0.9, 0.1, 0.0, 1.0
        a, b = (lower-mean)/std, (upper-mean)/std
        model.ecc.prior = Prior(scipy.stats.truncnorm(loc=mean, scale=std,
                                                      a=a, b=b))
        """
        self._rv = rv
        pass

    def pdf(self, value):
        # The astype() calls prevent unsafe cast messages
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.pdf(v)

    def logpdf(self, value):
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.logpdf(v)


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


def UniformUnnormedRV(lower=-np.inf, upper=np.inf):
    r"""An unnormalized uniform prior suitable for unbounded or
    half-bounded intervals.

    : param lower : lower bound of parameter range
    : type lower : float
    : param upper : upper bound of parameter range
    : type upper : float
    : return : a frozen rv_continuous instance with pdf equal to unity
               in the allowed interval and 0 elsewhere
    """
    return _UniformUnnormedRV_generator(name='unnormed_uniform',
                                        a=lower, b=upper)


def UniformBoundedRV(lower=0., upper=1.):
    r"""A uniform prior between two finite bounds.
    This is a convenience function with more natural bound parameters
    than `scipy.stats.uniform`.

    : param lower : lower bound of parameter range
    : type lower : float
    : param upper : upper bound of parameter range
    : type upper : float
    : return : a frozen rv_continuous instance with normalized uniform
               probability inside the range [lower, upper] and 0 outside
    """
    uu = scipy.stats.uniform(lower, (upper - lower))
    return uu
