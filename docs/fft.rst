
.. module:: enterprise
   :noindex:

.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <_static/notebooks/mdc.ipynb>`_.

.. _mdc:

Red noise modeling
=======================

In the beginning of Enterprise red noise modeling, the red noise prior was
always modeled as a diagonal matrix, meaning that the Fourier coefficients
were assumed to be uncorrelated. This model was introduced by Lentati et al.
(2013), and explained by van Haasteren and Vallisneri (2014). In practice this
has been a good-enough approximation, but it is not exact.

As of early 2025 we now have a more realistic implementation of red noise
priors that include correlations between the basis functions. The `FFT`
method as it is called is a rank-reduced time-domain implementation, meaning
it does not rely on Fourier modes, but on regularly sampled coarse grained
time samples. Here we briefly explain how to use it.



Red noise modeling
-------------------

The traditional old-style way of modeling was done like:

.. code:: python

    rn_pl = powerlaw(log10_A=rn_log10_A, gamma=rn_gamma)
    rn_phi = gp_signals.FourierBasisGP(spectrum=rn_pl, components=n_components, Tspan=Tspan)

For the FFT time-domain model, one would do:

.. code:: python

    rn_pl = powerlaw(log10_A=rn_log10_A, gamma=rn_gamma)
    rn_fft = gp_signals.FFTBasisGP(spectrum=rn_pl, components=n_components, oversample=3, cutoff=3)

The same spectral function can be used. Free spectrum is NOT supported yet.
Instead of `components`, we can also pass `knots=`, where it is understood that
`knots=2*n_components+1`. This is because `components` actually means
frequencies.  In the time-domain, the number of `knots` is the number of
`modes+1`, because we cannot just omit the DC term.

The `oversample` parameter determines how densely the PSD is sampled in
frequencies. With `oversample=1` we would use frequencies at spacing of
`df=1/T`.  With `oversample=3` (the default), the frequency spacing is
`df=1/(3T)`. Note that this is a way to numerically approximate the
Wiener-Khinchin integral. With oversample sufficiently large, the FFT is an
excellent approximation of the analytical integral. For powerlaw signals,
`oversample=3` seems a very reasonable choice.

The `cutoff` parameter is used to specify below which frequency `fcut = 1 /
(cutoff*Tspan)` we set the PSD equal to zero. Note that this parameterization
(which is also in Discovery) is a bit ambiguous, as fcut may not correspond to
an actual bin of the FFT: especially if oversample is not a high number this
can cause a mismatch. In case of a mismatch, `fcut` is rounded up to the next
oversampled-FFT frequency bin. Instead of `cutoff`, the parameter `cutbins` can
also be used (this overrides cutoff). With cutbins the low-frequency cutoff is
set at: `fcut = cutbins / (oversample * Tspan)`, and its interpretation is less
ambiguous: it is the number of bins of the over-sampled PSD of the FFT that is
being zeroed out.

Common signals
--------------

For common signals, instead of:

.. code:: python

    gw_pl = powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
    orf = utils.hd_orf()
    crn_phi = gp_signals.FourierBasisCommonGP(gw_pl, orf, components=20, name='gw', Tspan=Tspan)


one would do:

.. code:: python

    gw_pl = powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
    orf = utils.hd_orf()
    crn_fft = gp_signals.FFTBasisCommonGP(gw_pl, orf, components=20, name='gw', Tspan=Tspan, start_time=start_time)

Chromatic signals
-----------------

DM-variations and Chromatic noise can be similarly set up:

.. code:: python

    nknots = 81
    dm_basis = utils.create_fft_time_basis_dm(nknots=nknots)
    dm_pl = powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
    dm_fft = gp_signals.FFTBasisGP(dm_pl, basis=dm_basis, nknots=nknots, name='dmgp')

    chrom_basis = utils.create_fft_time_basis_chromatic(nknots=nknots, idx=chrom_idx)
    chrom_pl = powerlaw(log10_A=chrom_log10_A, gamma=chrom_gamma)
    chrom_fft = gp_signals.FFTBasisGP(chrom_pl, basis=chrom_basis, nknots=nknots, name='chromgp')
