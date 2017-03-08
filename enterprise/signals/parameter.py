# parameter.py

# Defines Parameter class for timing model parameters

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from enterprise.signals import prior


class Parameter(object):
    """Class describing a single signal parameter.

    The ``Parameter`` class will be initalized with several options

    :param name: The formal name of the parameter.
    :param value: The initial value of the parameter.
    :param description: A short description of the parameter.
    :param uncertainty: An initial approximate uncertainty.
    :param vary: Boolean flag indicating that the parameter will vary.
    :param prior: Initalized instance of ``Prior`` class.

    .. note:: This class is still under construction.
        As ``enerprise`` develops this class will likely grow and change.

    """

    def __init__(self, name=None, value=None, description=None,
                 uncertainty=None, vary=True,
                 prior=prior.Prior(prior.UniformUnnormedRV())):

        self.name = name
        self.value = value  # TODO: may want to make this private
        self.description = description
        self.uncertainty = uncertainty  # TODO: may want to make this private
        self.vary = vary
        self.prior = prior

    @property
    def prior(self):
        """Return prior instance."""
        return self._prior

    @prior.setter
    def prior(self, p):
        """Set prior instance."""
        if not isinstance(p, prior.Prior):
            # TODO: should use error logging
            msg = 'ERROR: prior must be instance of Prior().'
            raise ValueError(msg)
        self._prior = p

    def prior_pdf(self, value=None, logpdf=True):
        """Return the prior probability.

        The prior pdf can be evaluated at the current stored value or the
        proposed value. If ``value`` is passed as None then the stored value
        will be used.

        :param value: Proposed parameter value [default=None].
        :param logpdf: Return log prior [default=True].
        :return: prior probability value

        """
        val = self.value if value is None else value
        if logpdf:
            return self.prior.logpdf(val)
        else:
            return self.prior.pdf(val)
