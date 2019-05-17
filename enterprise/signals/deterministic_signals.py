# deterministic_signals.py
"""Contains class factories for deterministic signals.
Determinisitc signals are defined as the class of signals that have a
delay that is to be subtracted from the residuals.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from enterprise import pulsar
from enterprise.signals import signal_base
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import utils
from enterprise.signals.selections import Selection


def Deterministic(waveform, selection=Selection(selections.no_selection),
                  name=''):
    """Class factory for generic deterministic signals."""

    class Deterministic(signal_base.Signal):
        signal_type = 'deterministic'
        signal_name = name
        signal_id = name

        def __init__(self, psr):
            super(Deterministic, self).__init__(psr)
            self.name = self.psrname + '_' + self.signal_id
            self._do_selection(psr, waveform, selection)

        def _do_selection(self, psr, waveform, selection):

            sel = selection(psr)
            self._keys = list(sorted(sel.masks.keys()))
            self._masks = [sel.masks[key] for key in self._keys]
            self._delay = np.zeros(len(psr.toas))
            self._wf, self._params = {}, {}
            for key, mask in zip(self._keys, self._masks):
                pnames = [psr.name, name, key]
                pname = '_'.join([n for n in pnames if n])
                self._wf[key] = waveform(pname, psr=psr)
                params = self._wf[key]._params.values()
                for param in params:
                    self._params[param.name] = param

        @property
        def delay_params(self):
            """Get any varying ndiag parameters."""
            return [pp.name for pp in self.params]

        @signal_base.cache_call('delay_params')
        def get_delay(self, params):
            """Return signal delay."""
            for key, mask in zip(self._keys, self._masks):
                self._delay[mask] = self._wf[key](params=params, mask=mask)
            return self._delay

    return Deterministic


def PhysicalEphemerisSignal(
    frame_drift_rate=parameter.Uniform(-1e-9, 1e-9)('frame_drift_rate'),
    d_jupiter_mass=parameter.Normal(0, 1.54976690e-11)('d_jupiter_mass'),
    d_saturn_mass=parameter.Normal(0, 8.17306184e-12)('d_saturn_mass'),
    d_uranus_mass=parameter.Normal(0, 5.71923361e-11)('d_uranus_mass'),
    d_neptune_mass=parameter.Normal(0, 7.96103855e-11)('d_neptune_mass'),
    jup_orb_elements=parameter.Uniform(-0.05,0.05,size=6)('jup_orb_elements'),
    sat_orb_elements=parameter.Uniform(-0.5,0.5,size=6)('sat_orb_elements'),
    jup_dyn_orb_elements=parameter.Uniform(-1e-4,1e-4,
                                           size=6)('jup_orb_elements'),
    jupsun_dyn_orb_elements=parameter.Uniform(-1e-4,1e-4,
                                          size=12)('jupsun_orb_elements'),
    dynamic=False, inc_jupiter_orb=True, inc_saturn_orb=False,
    inc_jupitersun_orb=False, use_epoch_toas=True,
    name=''):  # noqa: E125,E501

    """
    Class factory for physical ephemeris model signal.

    This function implements a physically motivated ephemeris delay model.
    It is parameterized by an overall frame drift rate, masses of gas giants,
    6 orbital elements of Jupiter (uses a PCA basis), and 6 orbital elements
    of Saturn (uses PCA basis).

    .. note:: This signal is only compatible with a tempo2 Pulsar object.

    The user can implement their own priors but we have set reasonable
    defaults.

    :param frame_drift_rate:
        ecliptic z-drift rate in units of rad/year referred to offset 1/1/2010.
        Default prior is Uniform(-1e-9, 1e-9).

    :param d_jupiter_mass:
        Mass deviation of jupiter in solar masses. Default prior taken from
        IAU mass measurement uncertainty - Normal(0, 1.54976690e-11)

    :param d_saturn_mass:
        Mass deviation of saturn in solar masses. Default prior taken from
        IAU mass measurement uncertainty - Normal(0, 8.17306184e-12)

    :param d_uranus_mass:
        Mass deviation of uranus in solar masses. Default prior taken from
        IAU mass measurement uncertainty - Normal(0, 5.71923361e-11)

    :param d_neptune_mass:
        Mass deviation of neptune in solar masses. Default prior taken from
        IAU mass measurement uncertainty - Normal(0, 7.96103855e-11)

    :param jup_orb_elements:
        Amplitudes of PCA basis of jupiter orbital elements including
        (1) semi-major axis
        (2) eccentricity
        (3) inclination
        (4) longitude of the ascending node
        (5) longitude of perihelion
        (6) mean longitude

        Default prior is Uniform(-0.05, 0.05) for each element.

    :param sat_orb_elements:
        Amplitudes of PCA basis of saturn orbital elements including
        (1) semi-major axis
        (2) eccentricity
        (3) inclination
        (4) longitude of the ascending node
        (5) longitude of perihelion
        (6) mean longitude

        Default prior is Uniform(-0.5, 0.5) for each element.

    :param jup_dyn_orb_elements:
        Amplitudes of PCA basis of jupiter dynamic position and velocity.

        Default prior is Uniform(-1e-4, 1e-4) for each element.

    :param jupsun_dyn_orb_elements:
        Amplitudes of PCA basis of jupiter and Sun dynamic position and
        velocity

        Default prior is Uniform(-1e-4, 1e-4) for each element.

    :param inc_jupiter_orb:
        Boolean indicating whether or not to include jupiter's orbital
        elements as free parameters in model. Default: True

    :param inc_saturn_orb:
        Boolean indicating whether or not to include saturn's orbital
        elements as free parameters in model. Default: False

    :param inc_jupitersun_orb:
        Boolean indicating whether or not to include Jupiter's and Sun's
        orbital elements as free parameters in model. Default: False

    :param dynamic:
        Boolean indicating whether or not to model the physical ephemeris
        perturbation using dynamic partials. Default: False

    :param use_epoch_toas:
        Use interpolation from epoch to full TOAs. This option reduces
        computational cost for large multi-channel TOA data sets.
        Default: True
    """

    # turn off jupiter orbital element parameters if not including in signal
    if not inc_jupiter_orb:
        jup_orb_elements = np.zeros(6)
        jup_dyn_orb_elements = np.zeros(6)

    # turn off jupiter-sun orbital element parameters if
    # not including in signal
    if not inc_jupitersun_orb:
        jupsun_dyn_orb_elements = np.zeros(12)

    # turn off saturn orbital element parameters if not including in signal
    if not inc_saturn_orb:
        sat_orb_elements = np.zeros(6)

    # define waveform
    (jup_mjd, jup_orbelxyz, sat_mjd,
        sat_orbelxyz, jup_dyn_mjd,
        jup_dyn_orbelxyz, jupsun_dyn_mjd,
        jupsun_dyn_orbelxyz) = utils.get_planet_orbital_elements()
    wf = utils.physical_ephem_delay(frame_drift_rate=frame_drift_rate,
                                    d_jupiter_mass=d_jupiter_mass,
                                    d_saturn_mass=d_saturn_mass,
                                    d_uranus_mass=d_uranus_mass,
                                    d_neptune_mass=d_neptune_mass,
                                    dynamic=dynamic,
                                    jup_orb_elements=jup_orb_elements,
                                    jup_dyn_orb_elements=jup_dyn_orb_elements,
                                    jupsun_dyn_orb_elements=\
                                        jupsun_dyn_orb_elements,
                                    sat_orb_elements=sat_orb_elements,
                                    inc_jupiter_orb=inc_jupiter_orb,
                                    inc_jupitersun_orb=inc_jupitersun_orb,
                                    inc_saturn_orb=inc_saturn_orb,
                                    jup_orbelxyz=jup_orbelxyz,
                                    jup_mjd=jup_mjd,
                                    sat_orbelxyz=sat_orbelxyz,
                                    sat_mjd=sat_mjd,
                                    jup_dyn_orbelxyz=jup_dyn_orbelxyz,
                                    jup_dyn_mjd=jup_dyn_mjd,
                                    jupsun_dyn_orbelxyz=jupsun_dyn_orbelxyz,
                                    jupsun_dyn_mjd=jupsun_dyn_mjd)

    BaseClass = Deterministic(wf, name=name)

    class PhysicalEphemerisSignal(BaseClass):
        signal_name = 'phys_ephem'
        signal_id = 'phys_ephem_' + name if name else 'phys_ephem'

        def __init__(self, psr):

            # not available for PINT yet
            if isinstance(psr, pulsar.PintPulsar):
                msg = 'Physical Ephemeris model is not compatible with PINT '
                msg += 'at this time.'
                raise NotImplementedError(msg)

            super(PhysicalEphemerisSignal, self).__init__(psr)

            if use_epoch_toas:
                # get quantization matrix and calculate daily average TOAs
                U, _ = utils.create_quantization_matrix(psr.toas, nmin=1)
                self._uinds = utils.quant2ind(U)

                avetoas = np.array([psr.toas[sc].mean() for sc in self._uinds])
                self._avetoas = avetoas

                # interpolate ssb planet position vectors to avetoas
                planetssb = np.zeros((len(avetoas), 9, 3))
                for jj in range(9):
                    planetssb[:, jj, :] = np.array([
                        np.interp(avetoas, psr.toas, psr.planetssb[:,jj,aa])
                        for aa in range(3)]).T
                self._planetssb = planetssb

                # Interpolating the pulsar position vectors onto epoch TOAs
                pos_t = np.array([np.interp(avetoas, psr.toas, psr.pos_t[:,aa])
                                  for aa in range(3)]).T
                self._pos_t = pos_t

            # initialize delay
            self._delay = np.zeros(len(psr.toas))

        @signal_base.cache_call('delay_params')
        def get_delay(self, params):
            if use_epoch_toas:
                delay = self._wf[''](toas=self._avetoas,
                                     planetssb=self._planetssb,
                                     pos_t=self._pos_t,
                                     params=params)

                for slc, val in zip(self._uinds, delay):
                    self._delay[slc] = val
                return self._delay
            else:
                delay = self._wf[''](params=params)
                return delay

    return PhysicalEphemerisSignal
