Schur-complement likelihood
===========================

``SchurLogLikelihood`` is an opt-in, exact likelihood backend for PTA models
that contain common Gaussian-process basis signals, such as a stochastic
gravitational-wave background with Hellings--Downs correlations.  It separates
basis coefficients for individual pulsars from those of common signals.  It
eliminates the individual blocks before factoring the remaining matrix.

The signal model, parameter names, priors, and likelihood interface used by
samplers do not change.  Selecting this backend changes only how Enterprise
evaluates the marginalized likelihood.

The calculation reuses Enterprise's cached ``TNT`` and ``TNr`` products and
uses Cholesky solves for the individual and common blocks.  The standard
``LogLikelihood`` remains the default.

Schur complement
----------------

The standard coefficient matrix and right-hand side are

.. math::

   \Sigma = T^T N^{-1} T + \Phi^{-1}, \qquad d = T^T N^{-1} r.

For each pulsar, the combined basis is divided into individual columns
:math:`L_i` and common-signal columns :math:`C_i`.  Enterprise already knows
these column indices; the matrices below are extracted from ``get_TNT``,
``get_TNr``, and ``get_phiinv``.  When the prior separates these two sets,
reordering the coefficients gives

.. math::

   \Sigma = \begin{pmatrix} A & B \\ B^T & D \end{pmatrix},
   \qquad A = \operatorname{diag}(A_1,\ldots,A_P),

where

.. math::

   A_i &= (T_i^T N_i^{-1} T_i)_{L_i L_i} + (\Phi^{-1})_{L_i L_i}, \\
   B_i &= (T_i^T N_i^{-1} T_i)_{L_i C_i}, \\
   D &= (\Phi^{-1})_{CC}
        + \operatorname{diag}\bigl((T_i^T N_i^{-1} T_i)_{C_i C_i}\bigr).

Each :math:`A_i` is factorized independently.  Cholesky solves form the
smaller coupled matrix and its right-hand side:

.. math::

   S = D - B^T A^{-1} B, \qquad z = d_C - B^T A^{-1} d_L.

No inverse of :math:`A` or :math:`S` is formed.  The determinant and quadratic
form needed by the likelihood are

.. math::

   \log\det\Sigma &= \sum_i\log\det A_i + \log\det S, \\
   d^T\Sigma^{-1}d &= \sum_i d_{L_i}^T A_i^{-1}d_{L_i} + z^T S^{-1}z.

The full :math:`\Sigma` is neither assembled nor factorized.  Prior
covariances are evaluated through the existing ``PTA`` methods.

Coefficient axes are rescaled before Cholesky decomposition to reduce
differences in units.  The scaling is included in the solve and determinant.

Quick start
-----------

Reuse the pulsar models from an existing analysis and pass
``signal_base.SchurLogLikelihood`` to ``PTA``:

.. code-block:: python

   from enterprise.signals import signal_base

   pta = signal_base.PTA(
       pulsar_models, lnlikelihood=signal_base.SchurLogLikelihood
   )

   # Set any Constant parameters before the first likelihood evaluation.
   pta.set_default_params(noise_defaults)
   loglike = pta.get_lnlikelihood(params)

Omit ``lnlikelihood`` to use the standard backend.  No changes to the
pulsar models or sampler are needed.

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

``phiinv_method``
-----------------

``SchurLogLikelihood`` keeps this keyword in its call signature so existing
sampler code does not need to change:

.. code-block:: python

   loglike = pta.get_lnlikelihood(params)
   loglike = pta.get_lnlikelihood(params, phiinv_method="cliques")

``phiinv_method`` selects the existing ``PTA.get_phiinv`` calculation.  It is
passed through in both the Schur calculation and the standard fallback;
the likelihood backend is selected separately when constructing ``PTA``.

The inherited ``cholesky_sparse`` constructor option controls the standard
fallback.  Individual and reduced systems on the Schur path use dense
Cholesky factorizations.

Supported models
----------------

Both ``gp_signals.TimingModel()`` and
``gp_signals.MarginalizingTimingModel()`` are supported.  The calculation
uses the existing white-noise ``solve`` operations, including sparse, block,
Sherman--Morrison, and fast Sherman--Morrison ECORR with backend-dependent
white-noise parameters.

More than one type of common signal is supported.  For example, an HD process
and a monopole process can contribute to the same PTA; their covariance terms
are included through the existing prior calculation before the reduced system
is factored.  A pulsar can have only individual signals or no basis at all.

Models without common signals use the standard likelihood.  The standard
likelihood is also used when a dense prior couples individual and common
columns, or when a Schur Cholesky factorization fails.

Tests
-----

All 18 Schur tests pass, covering Feather, PINT, and tempo2 inputs.  The
tests compare the Schur result with the standard likelihood, full coefficient
calculations, and direct Gaussian covariances.  They cover both timing-model
paths, ECORR representations, multiple common signals, basis rescaling,
mixed priors, and parameter-vector inputs.

Run the likelihood tests with:

.. code-block:: console

   python -m pytest tests/test_likelihood.py

Performance
-----------

A PPTA31 benchmark uses 31 pulsars and all 114,202 TOAs, with HD correlations,
red and dispersion noise, backend-dependent white noise, and the additional
noise components and events in the per-pulsar models.  The full coefficient
matrix has dimension 32,698.  Schur separates it into 31 individual blocks
and a common block of dimension 1,860.

.. list-table:: Median likelihood evaluation time
   :header-rows: 1

   * - Varying parameters
     - Standard sparse
     - Schur
     - Speedup
   * - 194 search parameters, including events
     - 13.078 s
     - 5.431 s
     - 2.41x
   * - 172 covariance parameters, fixed events
     - 13.169 s
     - 5.357 s
     - 2.46x

Each entry uses 12 evaluations after initialization on the same Atlas node,
with one BLAS/OpenMP thread and identical parameter trajectories.  White-noise
parameters are fixed in this benchmark.  The largest absolute difference in
log likelihood is ``6.98e-10``.

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
   np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
