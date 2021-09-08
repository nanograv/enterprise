# pulsar.py
"""Class containing pulsar data from timing package [tempo2/PINT].
"""

import json
import logging
import os
import pickle

import astropy.constants as const
import astropy.units as u
import numpy as np
from ephem import Ecliptic, Equatorial

import enterprise
from enterprise.signals import utils

from enterprise.pulsar_inflate import PulsarInflater

logger = logging.getLogger(__name__)

try:
    import libstempo as t2
except ImportError:
    logger.warning("libstempo not installed. Will use PINT instead.")  # pragma: no cover
    t2 = None

try:
    import pint
    from pint.models import TimingModel, get_model_and_toas
    from pint.residuals import Residuals as resids
    from pint.toa import TOAs
except ImportError:
    logger.warning("PINT not installed. Will use libstempo instead.")  # pragma: no cover
    pint = None

if pint is None and t2 is None:
    err_msg = "Must have either PINT or libstempo timing package installed"
    raise ImportError(err_msg)


def get_maxobs(timfile):
    """Utility function to return number of lines in tim file.
    :param timfile:
        Full path to tim-file. For tim-files that use INCLUDEs this
        should be the base tim file.
    :returns: Number of lines in tim-file
    """

    maxobs = 0
    with open(timfile) as tfile:
        flines = tfile.readlines()
        lines = [ln for ln in flines if not ln.startswith("C")]
        if any(["INCLUDE" in ln for ln in lines]):
            for line in [ln for ln in lines if "INCLUDE" in ln]:
                maxobs += get_maxobs(line.split()[-1])
        else:
            maxobs = sum(1 for line in lines if line.rstrip("\n"))
    return maxobs


class BasePulsar(object):
    """Abstract Base Class for Pulsar objects."""

    def _get_pdist(self):
        dfile = enterprise.__path__[0] + "/datafiles/pulsar_distances.json"
        with open(dfile, "r") as fl:
            pdict = json.load(fl)

        if self.name[0] not in ["J", "B"]:
            if "J" + self.name in pdict:
                pdist = tuple(pdict.get("J" + self.name))
            else:
                pdist = tuple(pdict.get("B" + self.name, (1.0, 0.2)))
        else:
            pdist = tuple(pdict.get(self.name, (1.0, 0.2)))

        if pdist == (1.0, 0.2):
            msg = "WARNING: Could not find pulsar distance for "
            msg += "PSR {0}.".format(self.name)
            msg += " Setting value to 1 with 20% uncertainty."
            logger.warning(msg)
        return pdist

    def _get_radec_from_ecliptic(self, elong, elat):
        # convert via pyephem
        try:
            ec = Ecliptic(elong, elat)

            # check for B name
            if "B" in self.name:
                epoch = "1950"
            else:
                epoch = "2000"
            eq = Equatorial(ec, epoch=str(epoch))
            raj = np.double(eq.ra)
            decj = np.double(eq.dec)

        except TypeError:
            msg = "WARNING: Cannot fine sky location coordinates "
            msg += "for PSR {0}. ".format(self.name)
            msg += "Setting values to 0.0"
            logger.warning(msg)
            raj = 0.0
            decj = 0.0

        return raj, decj

    def _get_pos(self):
        return np.array(
            [np.cos(self._raj) * np.cos(self._decj), np.sin(self._raj) * np.cos(self._decj), np.sin(self._decj)]
        )

    def sort_data(self):
        """Sort data by time."""
        if self._sort:
            self._isort = np.argsort(self._toas, kind="mergesort")
            self._iisort = np.zeros(len(self._isort), dtype=int)
            for ii, p in enumerate(self._isort):
                self._iisort[p] = ii
        else:
            self._isort = slice(None, None, None)
            self._iisort = slice(None, None, None)

    def filter_data(self, start_time=None, end_time=None):
        """Filter data to create a time-slice of overall dataset."""
        if start_time is None and end_time is None:
            mask = np.ones(self._toas.shape, dtype=bool)
        else:
            mask = np.logical_and(self._toas >= start_time * 86400, self._toas <= end_time * 86400)

        self._toas = self._toas[mask]
        self._toaerrs = self._toaerrs[mask]
        self._residuals = self._residuals[mask]
        self._ssbfreqs = self._ssbfreqs[mask]

        self._designmatrix = self._designmatrix[mask, :]
        dmx_mask = np.sum(self._designmatrix, axis=0) != 0.0
        self._designmatrix = self._designmatrix[:, dmx_mask]

        if isinstance(self._flags, np.ndarray):
            self._flags = self._flags[mask]
        else:
            for key in self._flags:
                self._flags[key] = self._flags[key][mask]

        if self._planetssb is not None:
            self._planetssb = self.planetssb[mask, :, :]

        self.sort_data()

    def to_pickle(self, outdir=None):
        """Save object to pickle file."""

        # drop t2pulsar object
        if hasattr(self, "t2pulsar"):
            del self.t2pulsar
            msg = "t2pulsar object cannot be pickled and has been removed."
            logger.warning(msg)

        if hasattr(self, "pint_toas"):
            del self.pint_toas
            del self.model
            msg = "pint_toas and model objects cannot be pickled and have been removed."
            logger.warning(msg)

        if outdir is None:
            outdir = os.getcwd()

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(outdir + "/{0}.pkl".format(self.name), "wb") as f:
            pickle.dump(self, f)

    @property
    def isort(self):
        """Return sorting indices."""
        return self._isort

    @property
    def iisort(self):
        """Return inverse of sorting indices."""
        return self._iisort

    @property
    def toas(self):
        """Return array of TOAs in seconds."""
        return self._toas[self._isort]

    @property
    def stoas(self):
        """Return array of observatory TOAs in seconds."""
        return self._stoas[self._isort]

    @property
    def residuals(self):
        """Return array of residuals in seconds."""
        return self._residuals[self._isort]

    @property
    def toaerrs(self):
        """Return array of TOA errors in seconds."""
        return self._toaerrs[self._isort]

    @property
    def freqs(self):
        """Return array of radio frequencies in MHz."""
        return self._ssbfreqs[self._isort]

    @property
    def Mmat(self):
        """Return ntoa x npar design matrix."""
        return self._designmatrix[self._isort, :]

    @property
    def pdist(self):
        """Return tuple of pulsar distance and uncertainty in kpc."""
        return self._pdist

    @property
    def dm(self):
        """Return DM parameter from parfile."""
        return self._dm

    @property
    def dmx(self):
        """Return a dictionary of DMX-parameter values and stoa ranges
        from parfile."""
        return self._dmx

    @property
    def flags(self):
        """Return a dictionary of tim-file flags."""

        flagnames = self._flags.dtype.names if isinstance(self._flags, np.ndarray) else self._flags.keys()

        return {flag: self._flags[flag][self._isort] for flag in flagnames}

    @property
    def backend_flags(self):
        """Return array of backend flags.

        Not all TOAs have the same flags for all data sets. In order to
        facilitate this we have a ranked ordering system that will look
        for flags. The order is `group`, `g`, `sys`, `i`, `f`, `fe`+`be`.

        """

        # collect flag names
        flagnames = self._flags.dtype.names if isinstance(self._flags, np.ndarray) else list(self._flags.keys())

        # allocate array with widest dtype
        ret = np.zeros(len(self._toas), dtype=max([self._flags[name].dtype for name in flagnames]))

        # go through the flags in reverse order of preference
        # setting or replacing values for each TOA

        if "fe" in flagnames and "be" in flagnames:
            ret[:] = [(a + "_" + b if (a and b) else "") for a, b in zip(self._flags["fe"], self._flags["be"])]

        for flag in ["f", "i", "sys", "g", "group"]:
            if flag in flagnames:
                ret[:] = np.where(self._flags[flag] == "", ret, self._flags[flag])

        return ret

    @property
    def theta(self):
        """Return polar angle of pulsar in radians."""
        return np.pi / 2 - self._decj

    @property
    def phi(self):
        """Return azimuthal angle of pulsar in radians."""
        return self._raj

    @property
    def pos(self):
        """Return unit vector to pulsar."""
        return self._pos

    @property
    def pos_t(self):
        """Return unit vector to pulsar as function of time."""
        return self._pos_t[self._isort, :]

    @property
    def planetssb(self):
        """Return planetary position vectors at all timestamps"""
        return self._planetssb[self._isort, :, :]

    @property
    def sunssb(self):
        """Return sun position vector at all timestamps"""
        return self._sunssb[self._isort, :]


class PintPulsar(BasePulsar):
    def __init__(self, toas, model, sort=True, drop_pintpsr=True, planets=True):

        self._sort = sort
        self.planets = planets
        self.name = model.PSR.value
        if not drop_pintpsr:
            self.model = model
            self.pint_toas = toas

        self._toas = np.array(toas.table["tdbld"], dtype="float64") * 86400
        # saving also stoas (e.g., for DMX comparisons)
        self._stoas = np.array(toas.get_mjds().value, dtype="float64") * 86400
        self._residuals = np.array(resids(toas, model).time_resids.to(u.s), dtype="float64")
        self._toaerrs = np.array(toas.get_errors().to(u.s), dtype="float64")
        self._designmatrix = model.designmatrix(toas)[0]
        self._ssbfreqs = np.array(model.barycentric_radio_freq(toas), dtype="float64")

        # fitted parameters
        self.fitpars = ["Offset"] + [par for par in model.params if not getattr(model, par).frozen]

        # gather DM/DMX information if available
        self._set_dm(model)

        # set parameters
        spars = [par for par in model.params]
        self.setpars = [sp for sp in spars if sp not in self.fitpars]

        self._flags = {}
        for ii, obsflags in enumerate(toas.get_flags()):
            for jj, flag in enumerate(obsflags):

                if flag not in list(self._flags.keys()):
                    self._flags[flag] = [""] * toas.ntoas

                self._flags[flag][ii] = obsflags[flag]

        # convert flags to arrays
        # TODO probably better way to do this
        for key, val in self._flags.items():
            if isinstance(val[0], u.quantity.Quantity):
                self._flags[key] = np.array([v.value for v in val])
            else:
                self._flags[key] = np.array(val)

        self._pdist = self._get_pdist()
        self._raj, self._decj = self._get_radec(model)
        self._pos = self._get_pos()
        self._planetssb = self._get_planetssb(toas, model)
        self._sunssb = self._get_sunssb(toas, model)

        # TODO: pos_t not currently implemented
        self._pos_t = np.zeros((len(self._toas), 3))

        self.sort_data()

    def _set_dm(self, model):
        pars = [par for par in model.params if not getattr(model, par).frozen]

        if hasattr(model, "DM"):
            self._dm = model["DM"].value

        dmx = {
            par: {
                "DMX": model[par].value,
                "DMXerr": model[par].uncertainty_value,
                "DMXR1": model[par[:3] + "R1" + par[3:]].value,
                "DMXR2": model[par[:3] + "R2" + par[3:]].value,
                "fit": par in pars,
            }
            for par in pars
            if "DMX_" in par
        }

        if dmx:
            self._dmx = dmx

    def _get_radec(self, model):
        if hasattr(model, "RAJ") and hasattr(model, "DECJ"):
            return (model.RAJ.value, model.DECJ.value)
        else:
            # TODO: better way of dealing with units
            d2r = np.pi / 180
            elong, elat = model.ELONG.value, model.ELAT.value
            return self._get_radec_from_ecliptic(elong * d2r, elat * d2r)

    def _get_ssb_lsec(self, toas, obs_planet):
        """Get the planet to SSB vector in lightseconds from Pint table"""
        vec = toas.table[obs_planet] + toas.table["ssb_obs_pos"]
        return (vec / const.c).to("s").value

    def _get_planetssb(self, toas, model):
        planetssb = None
        if self.planets:
            planetssb = np.zeros((len(self._toas), 9, 6))
            # planetssb[:, 0, :] = self.t2pulsar.mercury_ssb
            # planetssb[:, 1, :] = self.t2pulsar.venus_ssb
            planetssb[:, 2, :3] = self._get_ssb_lsec(toas, "obs_earth_pos")
            # planetssb[:, 3, :] = self.t2pulsar.mars_ssb
            planetssb[:, 4, :3] = self._get_ssb_lsec(toas, "obs_jupiter_pos")
            planetssb[:, 5, :3] = self._get_ssb_lsec(toas, "obs_saturn_pos")
            planetssb[:, 6, :3] = self._get_ssb_lsec(toas, "obs_uranus_pos")
            planetssb[:, 7, :3] = self._get_ssb_lsec(toas, "obs_neptune_pos")
            # planetssb[:, 8, :] = self.t2pulsar.pluto_ssb

            # if hasattr(model, "ELAT") and hasattr(model, "ELONG"):
            #     for ii in range(9):
            #         planetssb[:, ii, :3] = utils.ecl2eq_vec(planetssb[:, ii, :3])
            #         # planetssb[:, ii, 3:] = utils.ecl2eq_vec(planetssb[:, ii, 3:])
        return planetssb

    def _get_sunssb(self, toas, model):
        sunssb = None
        if self.planets:
            sunssb = np.zeros((len(self._toas), 6))
            sunssb[:, :3] = self._get_ssb_lsec(toas, "obs_sun_pos")

            # if hasattr(model, "ELAT") and hasattr(model, "ELONG"):
            #     sunssb[:, :3] = utils.ecl2eq_vec(sunssb[:, :3])
            # #     sunssb[:, 3:] = utils.ecl2eq_vec(sunssb[:, 3:])
        return sunssb


class Tempo2Pulsar(BasePulsar):
    def __init__(self, t2pulsar, sort=True, drop_t2pulsar=True, planets=True):

        self._sort = sort
        self.t2pulsar = t2pulsar
        self.planets = planets
        self.name = str(t2pulsar.name)

        self._toas = np.double(t2pulsar.toas()) * 86400
        # saving also stoas (e.g., for DMX comparisons)
        self._stoas = np.double(t2pulsar.stoas) * 86400
        self._residuals = np.double(t2pulsar.residuals())
        self._toaerrs = np.double(t2pulsar.toaerrs) * 1e-6
        self._designmatrix = np.double(t2pulsar.designmatrix())
        self._ssbfreqs = np.double(t2pulsar.ssbfreqs()) / 1e6

        # fitted parameters
        self.fitpars = ["Offset"] + [str(p) for p in t2pulsar.pars()]

        # set parameters
        spars = [str(p) for p in t2pulsar.pars(which="set")]
        self.setpars = [sp for sp in spars if sp not in self.fitpars]

        flags = {}
        for key in t2pulsar.flags():
            flags[key] = t2pulsar.flagvals(key)

        # new-style storage of flags as a numpy record array (previously, psr._flags = flags)
        self._flags = np.zeros(len(self._toas), dtype=[(key, val.dtype) for key, val in flags.items()])
        for key, val in flags.items():
            self._flags[key] = val

        self._pdist = self._get_pdist()
        self._raj, self._decj = self._get_radec(t2pulsar)
        self._pos = self._get_pos()
        self._planetssb = self._get_planetssb(t2pulsar)
        self._sunssb = self._get_sunssb(t2pulsar)

        # gather DM/DMX information if available
        self._set_dm(t2pulsar)

        self._pos_t = t2pulsar.psrPos.copy()
        if "ELONG" and "ELAT" in np.concatenate((t2pulsar.pars(which="fit"), t2pulsar.pars(which="set"))):
            self._pos_t = utils.ecl2eq_vec(self._pos_t)

        self.sort_data()

        if drop_t2pulsar:
            del self.t2pulsar

    # gather DM/DMX information if available
    def _set_dm(self, t2pulsar):
        pars = t2pulsar.pars(which="set")

        if "DM" in pars:
            self._dm = t2pulsar["DM"].val

        dmx = {
            par: {
                "DMX": t2pulsar[par].val,
                "DMXerr": t2pulsar[par].err,
                "DMXR1": t2pulsar[par[:3] + "R1" + par[3:]].val,
                "DMXR2": t2pulsar[par[:3] + "R2" + par[3:]].val,
                "fit": par in pars,
            }
            for par in pars
            if "DMX_" in par
        }

        if dmx:
            self._dmx = dmx

    def _get_radec(self, t2pulsar):
        if "RAJ" in np.concatenate((t2pulsar.pars(which="fit"), t2pulsar.pars(which="set"))):
            return (np.double(t2pulsar["RAJ"].val), np.double(t2pulsar["DECJ"].val))

        else:
            # use ecliptic coordinates
            elong = t2pulsar["ELONG"].val
            elat = t2pulsar["ELAT"].val
            return self._get_radec_from_ecliptic(elong, elat)

    def _get_planetssb(self, t2pulsar):
        planetssb = None
        if self.planets:
            for ii in range(1, 10):
                tag = "DMASSPLANET" + str(ii)
                self.t2pulsar[tag].val = 0.0
            self.t2pulsar.formbats()
            planetssb = np.zeros((len(self._toas), 9, 6))
            planetssb[:, 0, :] = self.t2pulsar.mercury_ssb
            planetssb[:, 1, :] = self.t2pulsar.venus_ssb
            planetssb[:, 2, :] = self.t2pulsar.earth_ssb
            planetssb[:, 3, :] = self.t2pulsar.mars_ssb
            planetssb[:, 4, :] = self.t2pulsar.jupiter_ssb
            planetssb[:, 5, :] = self.t2pulsar.saturn_ssb
            planetssb[:, 6, :] = self.t2pulsar.uranus_ssb
            planetssb[:, 7, :] = self.t2pulsar.neptune_ssb
            planetssb[:, 8, :] = self.t2pulsar.pluto_ssb

            if "ELONG" and "ELAT" in np.concatenate((t2pulsar.pars(), t2pulsar.pars(which="set"))):
                for ii in range(9):
                    planetssb[:, ii, :3] = utils.ecl2eq_vec(planetssb[:, ii, :3])
                    planetssb[:, ii, 3:] = utils.ecl2eq_vec(planetssb[:, ii, 3:])
        return planetssb

    def _get_sunssb(self, t2pulsar):
        sunssb = None
        if self.planets:
            # for ii in range(1, 10):
            #     tag = 'DMASSPLANET' + str(ii)
            #     self.t2pulsar[tag].val = 0.0
            self.t2pulsar.formbats()
            sunssb = np.zeros((len(self._toas), 6))
            sunssb[:, :] = self.t2pulsar.sun_ssb

            if "ELONG" and "ELAT" in np.concatenate((t2pulsar.pars(), t2pulsar.pars(which="set"))):
                sunssb[:, :3] = utils.ecl2eq_vec(sunssb[:, :3])
                sunssb[:, 3:] = utils.ecl2eq_vec(sunssb[:, 3:])
        return sunssb

    # infrastructure for sharing Pulsar objects among processes
    # (currently Tempo2Pulsar only)
    # the Pulsar deflater will copy select numpy arrays to SharedMemory,
    # then replace them with pickleable objects that can be inflated
    # to numpy arrays with SharedMemory storage

    _todeflate = ["_designmatrix", "_planetssb", "_sunssb", "_flags"]
    _deflated = "pristine"

    def deflate(psr):  # pragma: py-lt-38
        if psr._deflated == "pristine":
            for attr in psr._todeflate:
                if isinstance(getattr(psr, attr), np.ndarray):
                    setattr(psr, attr, PulsarInflater(getattr(psr, attr)))

            psr._deflated = "deflated"

    def inflate(psr):  # pragma: py-lt-38
        if psr._deflated == "deflated":
            for attr in psr._todeflate:
                if isinstance(getattr(psr, attr), PulsarInflater):
                    setattr(psr, attr, getattr(psr, attr).inflate())

            psr._deflated = "inflated"

    def destroy(psr):  # pragma: py-lt-38
        if psr._deflated == "deflated":
            for attr in psr._todeflate:
                if isinstance(getattr(psr, attr), PulsarInflater):
                    getattr(psr, attr).destroy()

            psr._deflated = "destroyed"


def Pulsar(*args, **kwargs):
    ephem = kwargs.get("ephem", None)
    clk = kwargs.get("clk", None)
    bipm_version = kwargs.get("bipm_version", None)
    planets = kwargs.get("planets", True)
    sort = kwargs.get("sort", True)
    drop_t2pulsar = kwargs.get("drop_t2pulsar", True)
    drop_pintpsr = kwargs.get("drop_pintpsr", True)
    timing_package = kwargs.get("timing_package", "tempo2")

    if pint is not None:
        toas = [x for x in args if isinstance(x, TOAs)]
        model = [x for x in args if isinstance(x, TimingModel)]

    if t2 is not None:
        t2pulsar = [x for x in args if isinstance(x, t2.tempopulsar)]

    parfile = [x for x in args if isinstance(x, str) and x.split(".")[-1] == "par"]
    timfile = [x for x in args if isinstance(x, str) and x.split(".")[-1] in ["tim", "toa"]]

    if pint and toas and model:
        return PintPulsar(toas[0], model[0], sort=sort, drop_pintpsr=drop_pintpsr, planets=planets)
    elif t2 and t2pulsar:
        return Tempo2Pulsar(t2pulsar[0], sort=sort, drop_t2pulsar=drop_t2pulsar, planets=planets)
    elif parfile and timfile:
        # Check whether the two files exist
        if not os.path.isfile(parfile[0]) or not os.path.isfile(timfile[0]):
            msg = "Cannot find parfile {0} or timfile {1}!".format(parfile[0], timfile[0])
            raise IOError(msg)

        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(timfile[0])
        dirname = timfiletup[0] or "./"
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile[0], dirname)

        # get current directory
        cwd = os.getcwd()

        # Change directory to the base directory of the tim-file to deal with
        # INCLUDE statements in the tim-file
        os.chdir(dirname)

        if timing_package.lower() == "pint":
            if (clk is not None) and (bipm_version is None):
                bipm_version = clk.split("(")[1][:-1]
            model, toas = get_model_and_toas(
                relparfile, reltimfile, ephem=ephem, bipm_version=bipm_version, planets=planets
            )
            os.chdir(cwd)
            return PintPulsar(toas, model, sort=sort, drop_pintpsr=drop_pintpsr, planets=planets)

        elif timing_package.lower() == "tempo2":

            # hack to set maxobs
            maxobs = get_maxobs(reltimfile) + 100
            t2pulsar = t2.tempopulsar(relparfile, reltimfile, maxobs=maxobs, ephem=ephem, clk=clk)
            os.chdir(cwd)
            return Tempo2Pulsar(t2pulsar, sort=sort, drop_t2pulsar=drop_t2pulsar, planets=planets)

    raise ValueError("Unknown arguments {}".format(args))
