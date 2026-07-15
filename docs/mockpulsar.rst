.. _mockpulsar:

Mock data without PINT/tempo2
=============================

``MockPulsar`` builds an Enterprise-compatible pulsar object from plain
NumPy arrays. No timing package (PINT or tempo2/libstempo) is required at
runtime. Use it for injection studies, likelihood unit tests, and quick
pipeline prototypes.

What you get
------------

* TOAs, residuals, uncertainties, radio frequencies, flags, sky position
* A design matrix for :func:`~enterprise.signals.gp_signals.TimingModel`
  marginalization:

  * polynomial spindown basis ``Offset, Poly1, Poly2, …`` (years from
    ``pepoch_mjd``)
  * optional analytic astrometry ``RAJ, DECJ, PMRA, PMDEC, PX`` (PINT-compatible
    first derivatives; needs Astropy)

* Safe zero-filled ``planetssb`` / ``sunssb`` arrays so property access works

What you do **not** get
-----------------------

``MockPulsar`` is **not** a full timing model:

* Earth barycenter only (not observatory site), builtin solar-system ephemeris
* No Shapiro delay, binary model, DMX, FD, or JUMP columns
* Static ``pos_t`` (no proper-motion propagation of the unit vector)
* Astrometry columns are for *marginalization / basis geometry*, not for
  interpreting fitted RA/PM as catalog values without care

Supported Enterprise signals (typical)
--------------------------------------

* White noise (EFAC / EQUAD / ECORR with flags)
* ``TimingModel`` / ``MarginalizingTimingModel``
* Fourier red noise, DM noise, common-process GPs (e.g. GWB)

Avoid physical-ephemeris signals unless you fill ``planetssb`` yourself with
real vectors.

Minimal example
---------------

.. code-block:: python

    import numpy as np
    from enterprise.pulsar import MockPulsar
    from enterprise.signals import parameter, white_signals, gp_signals, utils
    from enterprise.signals.signal_base import PTA

    toas = np.linspace(53000.0, 58000.0, 200)
    # Optional: invent a J-name from coordinates
    # name = utils.get_psrname_from_pos(raj=raj, decj=decj)

    psr = MockPulsar(
        obs_times_mjd=toas,
        name="J1855+0939",
        raj=4.9533700839400492,   # rad
        decj=0.16848694562363042,  # rad
        # or: elong=..., elat=...  (degrees)
        freqs_mhz=1440.0,
        residuals=np.zeros_like(toas),
        toaerrs=1e-6,              # seconds
        flags={"f": np.array(["sys"] * len(toas))},
        telescope="GBT",
        spindown_order=2,
        inc_astrometry=True,
        posepoch_mjd=55500.0,
        pepoch_mjd=55500.0,
        dm=13.3,
        distance_kpc=1.2,
    )

    psr.set_residuals(np.random.normal(0.0, 1e-7, size=len(toas)))

    ef = white_signals.MeasurementNoise(efac=parameter.Constant(1.0))
    tm = gp_signals.TimingModel()
    pl = utils.powerlaw(
        log10_A=parameter.Constant(-15.0),
        gamma=parameter.Constant(13.0 / 3.0),
    )
    rn = gp_signals.FourierBasisGP(pl, components=10)
    pta = PTA([(ef + tm + rn)(psr)])
    print(pta.get_lnlikelihood({}))

Units (astrometry design matrix)
--------------------------------

When ``inc_astrometry=True`` the analytic columns match PINT’s convention:

* ``RAJ``, ``DECJ`` — seconds per radian
* ``PMRA``, ``PMDEC`` — seconds per (mas/yr), with ``PMRA = mu_alpha*``
  (includes ``cos(delta)``)
* ``PX`` — seconds per mas

Frequencies are in **MHz** (Enterprise convention). Pass ``freqs_mhz=``, not Hz.

API notes
---------

* ``name`` is required. Do not rely on auto-naming for production noise
  dictionaries; use :func:`~enterprise.signals.utils.get_psrname_from_pos`
  only as a helper when inventing mock identities.
* ``setpars`` is empty: every design-matrix column is treated as free for
  TimingModel.
* The legacy keyword ``ssbfreqs`` is accepted with a deprecation warning;
  prefer ``freqs_mhz``.
