Schur-complement likelihood
===========================

``SchurLogLikelihood`` is an opt-in, exact likelihood backend for PTA models
that contain common Gaussian-process basis signals, such as a stochastic
gravitational-wave background with Hellings--Downs correlations.  It separates
Fourier coefficients for individual pulsars from those of common signals.  It
eliminates the individual blocks before factoring the remaining matrix.

The signal model, parameter names, priors, and likelihood interface used by
samplers do not change.  Selecting this backend changes only how Enterprise
evaluates the marginalized likelihood.

The first version does not retain Schur decompositions or other numerical
factors between likelihood calls.

Quick start
-----------

Pass ``signal_base.SchurLogLikelihood`` through the existing ``PTA``
``lnlikelihood`` hook:

.. code-block:: python

   from enterprise.signals import gp_signals, parameter, signal_base, utils
   from enterprise.signals import white_signals

   tmin = min(psr.toas.min() for psr in psrs)
   tmax = max(psr.toas.max() for psr in psrs)
   Tspan = tmax - tmin

   timing = gp_signals.TimingModel()
   white = white_signals.MeasurementNoise(
       efac=parameter.Constant(1.0)
   )

   red_spectrum = utils.powerlaw(
       log10_A=parameter.Uniform(-18, -12),
       gamma=parameter.Uniform(0, 7),
   )
   red = gp_signals.FourierBasisGP(
       red_spectrum, components=30, Tspan=Tspan, name="red_noise"
   )

   gw_spectrum = utils.powerlaw(
       log10_A=parameter.Uniform(-18, -12),
       gamma=parameter.Uniform(0, 7),
   )
   hd = gp_signals.FourierBasisCommonGP(
       gw_spectrum,
       utils.hd_orf(),
       components=30,
       Tspan=Tspan,
       name="gw",
   )

   model = timing + white + red + hd
   pulsar_models = [model(psr) for psr in psrs]

   pta = signal_base.PTA(
       pulsar_models, lnlikelihood=signal_base.SchurLogLikelihood
   )

   # Set any Constant parameters before the first likelihood evaluation.
   pta.set_default_params(noise_defaults)
   loglike = pta.get_lnlikelihood(params)

Existing analyses normally need to change only the ``PTA`` construction line:

.. code-block:: python

   # Standard Enterprise backend
   pta = signal_base.PTA(pulsar_models)

   # Exact Schur backend
   pta = signal_base.PTA(
       pulsar_models, lnlikelihood=signal_base.SchurLogLikelihood
   )

Likelihood inputs and samplers
------------------------------

``get_lnlikelihood`` retains the standard Enterprise interface.  It accepts a
parameter dictionary:

.. code-block:: python

   loglike = pta.get_lnlikelihood(
       {
           "gw_log10_A": -14.5,
           "gw_gamma": 13.0 / 3.0,
           # Include the remaining varying PTA parameters here.
       }
   )

It also accepts a flat sampler vector in the ordering described by
``pta.param_names``.  Enterprise maps the vector to its parameter dictionary
internally, so existing sampler code can continue to use:

.. code-block:: python

   loglike = pta.get_lnlikelihood(x)

No-HD and no-common models
--------------------------

The backend does not require a Hellings--Downs process.  If the model contains
no common basis signal, it delegates the evaluation to the standard Enterprise
likelihood.  A local-noise-only model therefore does not fail and does not need
special calling code.

``phiinv_method``
-----------------

``SchurLogLikelihood`` keeps this keyword in its call signature so existing
sampler code does not need to change:

.. code-block:: python

   loglike = pta.get_lnlikelihood(params)
   loglike = pta.get_lnlikelihood(params, phiinv_method="cliques")

The ``phiinv_method`` argument does not select the likelihood backend.  The
Householder--Schur calculation constructs the common covariance directly and
therefore does not invert it using ``phiinv_method``.  If the model is delegated
to the standard likelihood, the keyword is forwarded unchanged and retains
its standard Enterprise meaning.

Compatibility and fallback
--------------------------

The Householder formulation needs the white-noise representation to implement
``sqrtsolve``.  This is available for the ordinary
``gp_signals.TimingModel()`` path used in the example above.

``gp_signals.MarginalizingTimingModel()`` does not currently provide the
square-root solve required by the Householder formulation and is therefore
evaluated by the standard Enterprise likelihood.

White-noise representations without this operation, including the ``block``
and ``sparse`` ECORR implementations, are evaluated by the standard Enterprise
likelihood instead.  Dense coefficient priors that cannot be separated into
local and common columns use the same fallback.  The result remains exact and
the calling interface is unchanged, but a fallback uses the standard
factorization rather than the Schur reduction.

More than one type of common signal is supported.  For example, an HD process
and a monopole process can contribute to the same PTA; their covariance terms
are accumulated in the common block before the reduced system is factored.

Checking against the standard backend
-------------------------------------

For a new model family, a small numerical comparison can be made before a long
run:

.. code-block:: python

   import numpy as np

   standard = signal_base.PTA(pulsar_models)
   schur = signal_base.PTA(
       pulsar_models, lnlikelihood=signal_base.SchurLogLikelihood
   )
   standard.set_default_params(noise_defaults)
   schur.set_default_params(noise_defaults)

   expected = standard.get_lnlikelihood(params)
   actual = schur.get_lnlikelihood(params)
   np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)

The Schur backend is an algebraic reorganization of the same marginalized
Gaussian likelihood; it is not an approximate likelihood.  The two
factorization orders can nevertheless produce small floating-point rounding
differences, so bitwise equality is not expected.
