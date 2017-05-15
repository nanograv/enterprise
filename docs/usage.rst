
.. module:: enterprise

.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <_static/notebooks/usage.ipynb>`_.

.. _usage:

Usage
=====


Setting up the ``Pulsar`` object
--------------------------------

``enterprise`` uses a specific ``Pulsar`` object to store all of the
relevant pulsar information (i.e. TOAs, residuals, error bars, flags,
etc) from the timing package. Eventually ``enterprise`` will support
both ``PINT`` and ``tempo2``; however, for the moment it only supports
``tempo2`` through the
```libstempo`` <https://github.com/vallis/libstempo>`__ package. This
object is then used to initalize ``Signal``\ s that define the
generative model for the pulsar residuals. This is in keeping with the
overall ``enterprise`` philosophy that the pulsar data should be as
loosley coupled as possible to the pulsar model.

Below we initialize a pulsar class with NANOGrav B1855+09 data by
passing it the par and tim file.

.. code:: python

    # pulsar file information
    parfiles = 'data/B1855+09_NANOGrav_11yv0.gls.par'
    timfiles = 'data/B1855+09_NANOGrav_11yv0.tim'
    
    psr = Pulsar(parfiles, timfiles)

Parameters
----------

In ``enterprise`` signal parameters are set by specifying a prior
distribution (i.e., Uniform, Normal, etc.). Below we will give an
example of this functionality.

.. code:: python

    # lets define an efac parameter with a uniform prior from [0.5, 5]
    efac = parameter.Uniform(0.5, 5)

This is an *abstract* parameter class in that it is not yet intialized.
It is equivalent to defining the class via the standard nomenclature
``class efac(object)...`` The parameter is then intialized via a name.
This way, a single parameter class can be initialized for multiple
signal parameters with different names (i.e. EFAC per observing backend,
etc). Once the parameter is initialized then you then have access to
many useful methods.

.. code:: python

    # initialize efac parameter with name "efac_1"
    efac1 = efac('efac_1')
    
    # return parameter name
    print(efac1.name)
    
    # get pdf at a point (log pdf is access)
    print(efac1.get_pdf(1.3), efac1.get_logpdf(1.3))
    
    # return 5 samples from this prior distribution
    print(efac1.sample(size=5))


.. parsed-literal::

    efac_1
    0.222222222222 -1.50407739678
    [ 2.94183156  2.51064331  2.44655821  1.1799445   2.41031989]


Set up a basic pulsar noise model
---------------------------------

For our basic noise model we will use standard EFAC, EQUAD, and ECORR
white noise with a powerlaw red noise parameterized by an amplitude and
spectral index. Using the methods described above we define our
parameters for our noise model below.

.. code:: python

    # white and red noise parameters with uniform priors
    efac = parameter.Uniform(0.5, 5)
    log10_equad = parameter.Uniform(-10, -5)
    log10_ecorr = parameter.Uniform(-10, -5)
    log10_A = parameter.Uniform(-18, -12)
    gamma = parameter.Uniform(1, 7)

White noise signals
~~~~~~~~~~~~~~~~~~~

White noise signals are straightforward to intialize

.. code:: python

    # EFAC, EQUAD, and ECORR signals
    ef = ws.MeasurementNoise(efac=efac)
    eq = ws.EquadNoise(log10_equad=log10_equad)
    ec = gs.EcorrBasisModel(log10_ecorr=log10_ecorr)

Again, these are *abstract* classes that will be in itialized when
passes a ``Pulsar`` object. This, again, makes for ease of use when
constucting pulsar signal models in that these classes are created on
the fly and can be re-intialized with different pulsars.

Red noise signals
~~~~~~~~~~~~~~~~~

Red noise signals are handled somewhat differently than other signals in
that we do not create the class by passing the parameters directly.
Instead we use the ``Function`` factory (creates a class, not an
instance) to set the red noise PSD used (i.e. powerlaw, spectrum,
broken, etc). This allows the user to define custom PSDs with no extra
coding overhead other than the PSD definition itself.

.. code:: python

    # Use Function object to set power-law red noise with uniform priors
    pl = Function(utils.powerlaw, log10_A=log10_A, gamma=gamma)
    
    # red noise signal using Fourier GP
    rn = gs.FourierBasisGP(spectrum=pl, components=30)

Here we have defined a power-law function class that will be initialized
when the red noise class is initialized. The red noise signal model is
then a powerlaw red noise process modeled via a Fourier basis Gaussian
Process with 30 components.

Linear timing model
~~~~~~~~~~~~~~~~~~~

We must include the timing model in all of our analyses. In this case we
treat it as a gaussian process with very large variances. Thus, this is
equvalent to marginalizing over the linear timing model coefficients
assuming uniform priors. In ``enterprise`` this is setup via:

.. code:: python

    # timing model as GP (no parameters)
    tm = gs.TimingModel()

Initializing the model
~~~~~~~~~~~~~~~~~~~~~~

Now that we have all of our signals defined we can define our total
noise model as the sum of all of the components and intialize by passing
that combined signal class the pulsar object. Is that awesome or what!

.. code:: python

    # create combined signal class with some metaclass magic
    s = ef + ec + eq + rn + tm
    
    # initialize model with pulsar object
    pm = s(psr)
    
    # print out the parameter names and priors
    pm.params




.. parsed-literal::

    ["B1855+09_efac":Uniform(0.5,5),
     "B1855+09_gamma":Uniform(1,7),
     "B1855+09_log10_A":Uniform(-18,-12),
     "B1855+09_log10_ecorr":Uniform(-10,-5),
     "B1855+09_log10_equad":Uniform(-10,-5)]


