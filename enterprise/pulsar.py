# pulsar.py
"""Class containing pulsar data from timing package [tempo2/PINT].
"""

import contextlib
import json
import logging
import os
import pickle

from pyarrow import feather
from pyarrow import Table
from io import StringIO

import numpy as np
from ephem import Ecliptic, Equatorial
from astropy.time import Time

import enterprise
from enterprise.signals import utils

from enterprise.pulsar_inflate import PulsarInflater

logger = logging.getLogger(__name__)

try:
    import libstempo as t2
except ImportError:
    logger.warning(
        "libstempo not installed. PINT or libstempo are required to use par and tim files."
    )  # pragma: no cover
    t2 = None

try:
    import pint
    from pint.models import TimingModel, get_model_and_toas
    from pint.residuals import Residuals as resids
    from pint.toa import TOAs
except ImportError:
    logger.warning("PINT not installed. PINT or libstempo are required to use par and tim files.")  # pragma: no cover
    pint = None

try:
    import astropy.constants as const
    import astropy.units as u
except ImportError:  # pragma: no cover
    const = None
    u = None


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
            msg = "WARNING: Cannot find sky location coordinates "
            msg += "for PSR {0}. ".format(self.name)
            msg += "Setting values to 0.0"
            logger.warning(msg)
            raj = 0.0
            decj = 0.0

        return raj, decj

    def _get_pos(self):
        return np.array(
            [
                np.cos(self._raj) * np.cos(self._decj),
                np.sin(self._raj) * np.cos(self._decj),
                np.sin(self._decj),
            ]
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

    def filter_data(self, mask=None, start_time=None, end_time=None):
        """
        Filters the dataset to create a time-slice based on a custom mask or time range.

        Parameters:
            mask (array-like, optional): Boolean array specifying which data to keep.
                                         If None, `start_time` and `end_time` are used. Default is None.
            start_time (float, optional): Start time (MJD) for filtering. Ignored if `mask` is provided. Default None.
            end_time (float, optional): End time (MJD) for filtering. Ignored if `mask` is provided. Default None.
        """

        start_time = start_time * 86400 if start_time is not None else np.min(self._toas)
        end_time = end_time * 86400 if end_time is not None else np.max(self._toas)
        mask = mask if mask is not None else np.logical_and(self._toas >= start_time, self._toas <= end_time)

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
            self._planetssb = self._planetssb[mask, :, :]

        self.sort_data()

    def to_feather(self, filename, noisedict=None):
        FeatherPulsar.save_feather(self, filename, noisedict=noisedict)

    def drop_not_picklable(self):
        """Drop all attributes that cannot be pickled.

        Derived classes should implement this if they have
        any such attributes.
        """
        pass

    def to_pickle(self, outdir=None):
        """Save object to pickle file."""

        self.drop_not_picklable()

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

    def set_flags(self, flagname, values):
        """Set value of existing or new flags."""

        if isinstance(self._flags, np.ndarray):
            raise NotImplementedError("Cannot set flags when stored as numpy.ndarray.")
        else:
            self._flags[flagname] = values[self._iisort]

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

        return ret[self._isort]

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
        """Return unit vector from SSB to pulsar at fiducial POSEPOCH."""
        return self._pos

    @property
    def pos_t(self):
        """Return unit vector from SSB to pulsar as function of time."""
        return self._pos_t[self._isort, :]

    @property
    def planetssb(self):
        """Return planetary position vectors at all timestamps"""
        return self._planetssb[self._isort, :, :]

    @property
    def sunssb(self):
        """Return sun position vector at all timestamps"""
        return self._sunssb[self._isort, :]

    @property
    def telescope(self):
        """Return telescope name at all timestamps"""
        return self._telescope[self._isort]


class PintPulsar(BasePulsar):
    def __init__(self, toas, model, sort=True, drop_pintpsr=True, planets=True):
        self._sort = sort
        self.planets = planets
        self.name = model.PSR.value

        if not drop_pintpsr:
            self.model = model
            self.parfile = model.as_parfile()
            self.pint_toas = toas
            with StringIO() as tim:
                toas.write_TOA_file(tim)
                self.timfile = tim.getvalue()

        # these are TDB but not barycentered
        # self._toas = np.array(toas.table["tdbld"], dtype="float64") * 86400
        self._toas = np.array(model.get_barycentric_toas(toas).value, dtype="float64") * 86400
        # saving also stoas (e.g., for DMX comparisons)
        self._stoas = np.array(toas.get_mjds().value, dtype="float64") * 86400
        self._residuals = np.array(resids(toas, model).time_resids.to(u.s), dtype="float64")
        self._toaerrs = np.array(toas.get_errors().to(u.s), dtype="float64")
        self._designmatrix, self.fitpars, self.designmatrix_units = model.designmatrix(toas)
        self._ssbfreqs = np.array(model.barycentric_radio_freq(toas), dtype="float64")
        self._telescope = np.array(toas.get_obss())

        # gather DM/DMX information if available
        self._set_dm(model)

        # set parameters
        self.setpars = [sp for sp in model.params if sp not in self.fitpars]

        # FIXME: this can be done more cleanly using PINT
        self._flags = {}
        for ii, obsflags in enumerate(toas.get_flags()):
            for jj, flag in enumerate(obsflags):
                if flag not in list(self._flags.keys()):
                    self._flags[flag] = [""] * toas.ntoas

                self._flags[flag][ii] = obsflags[flag]

        # convert flags to arrays
        # TODO probably better way to do this
        #      -- PINT always stores flags as strings
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

        which_astrometry = (
            "AstrometryEquatorial" if "AstrometryEquatorial" in model.components else "AstrometryEcliptic"
        )

        self._pos_t = (
            model.components[which_astrometry]
            .ssb_to_psb_xyz_ICRS(Time(model.get_barycentric_toas(toas), format="mjd"))
            .value
        )

        self.sort_data()

    def drop_pintpsr(self):
        with contextlib.suppress(NameError):
            del self.model
            del self.parfile
            del self.pint_toas
            del self.timfile

    def drop_not_picklable(self):
        with contextlib.suppress(AttributeError):
            del self.model
            del self.pint_toas
            logger.warning("pint_toas and model objects cannot be pickled and have been removed.")

        return super().drop_not_picklable()

    def _set_dm(self, model):
        pars = [par for par in model.params if not getattr(model, par).frozen]

        if hasattr(model, "DM"):
            self._dm = float(model["DM"].value)

        dmx = {
            par: {
                "DMX": float(model[par].value),
                "DMXerr": None if model[par].uncertainty_value is None else float(model[par].uncertainty_value),
                "DMXR1": float(model[par[:3] + "R1" + par[3:]].value),
                "DMXR2": float(model[par[:3] + "R2" + par[3:]].value),
                "fit": par in pars,
            }
            for par in pars
            if "DMX_" in par
        }

        if dmx:
            self._dmx = dmx
        else:
            self._dmx = None

    def _get_radec(self, model):
        if hasattr(model, "RAJ") and hasattr(model, "DECJ"):
            raj = model.RAJ.quantity.to(u.rad).value
            decj = model.DECJ.quantity.to(u.rad).value
            return raj, decj
        else:
            elong = model.ELONG.quantity.to(u.rad).value
            elat = model.ELAT.quantity.to(u.rad).value
            return self._get_radec_from_ecliptic(elong, elat)

    def _get_ssb_lsec(self, toas, obs_planet):
        """Get the planet to SSB vector in lightseconds from Pint table"""
        if obs_planet not in toas.table.colnames:
            err_msg = f"{obs_planet} is not in toas.table.colnames. Either "
            err_msg += "`planet` flag is not True  in `toas` or further Pint "
            err_msg += "development to add additional planets is needed."
            raise ValueError(err_msg)
        vec = toas.table[obs_planet] + toas.table["ssb_obs_pos"]
        return (vec / const.c).to("s").value

    def _get_planetssb(self, toas, model):
        planetssb = None
        """
        Currently Pint only has position vectors for:
        [Earth, Jupiter, Saturn, Uranus, Neptune]
        No velocity vectors available
        [Mercury, Venus, Mars, Pluto] unavailable pending Pint enhancements.
        """
        if self.planets:
            planetssb = np.empty((len(self._toas), 9, 6))
            planetssb[:] = np.nan
            planetssb[:, 2, :3] = self._get_ssb_lsec(toas, "obs_earth_pos")
            planetssb[:, 4, :3] = self._get_ssb_lsec(toas, "obs_jupiter_pos")
            planetssb[:, 5, :3] = self._get_ssb_lsec(toas, "obs_saturn_pos")
            planetssb[:, 6, :3] = self._get_ssb_lsec(toas, "obs_uranus_pos")
            planetssb[:, 7, :3] = self._get_ssb_lsec(toas, "obs_neptune_pos")

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
    def __init__(
        self,
        t2pulsar,
        sort=True,
        drop_t2pulsar=True,
        planets=True,
        par_name=None,
        tim_name=None,
    ):
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
        self._telescope = np.char.decode(t2pulsar.telescope(), encoding="ascii")

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
            self.drop_tempopsr()
        else:
            if par_name is not None and os.path.exists(par_name):
                self.parfile = open(par_name).read()
            if tim_name is not None and os.path.exists(tim_name):
                self.timfile = open(tim_name).read()

    def drop_tempopsr(self):
        with contextlib.suppress(NameError):
            del self.t2pulsar

    # gather DM/DMX information if available
    def _set_dm(self, t2pulsar):
        pars = t2pulsar.pars(which="set")

        if "DM" in pars:
            self._dm = float(t2pulsar["DM"].val)

        dmx = {
            par: {
                "DMX": float(t2pulsar[par].val),
                "DMXerr": float(t2pulsar[par].err),
                "DMXR1": float(t2pulsar[par[:3] + "R1" + par[3:]].val),
                "DMXR2": float(t2pulsar[par[:3] + "R2" + par[3:]].val),
                "fit": par in pars,
            }
            for par in pars
            if "DMX_" in par
        }

        if dmx:
            self._dmx = dmx
        else:
            self._dmx = None

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
            #     tag = 'DMASSPLANET' + str(ii)@pytest.mark.skipif(t2 is None, reason="TEMPO2/libstempo not available")
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

    def drop_not_picklable(self):
        with contextlib.suppress(AttributeError):
            del self.t2pulsar
            logger.warning("t2pulsar object cannot be pickled and has been removed.")
        return super().drop_not_picklable()

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


class FeatherPulsar:
    columns = ["toas", "stoas", "toaerrs", "residuals", "freqs", "backend_flags", "telescope"]
    vector_columns = ["Mmat", "sunssb", "pos_t"]
    tensor_columns = ["planetssb"]
    # flags are done separately
    metadata = ["name", "dm", "dmx", "pdist", "pos", "phi", "theta"]
    # notes: currently ignores _isort/__isort and gets sorted versions

    def __init__(self):
        pass

    def __str__(self):
        return f"<Pulsar {self.name}: {len(self.residuals)} res, {self.Mmat.shape[1]} pars>"

    def __repr__(self):
        return str(self)

    def sort_data(self):
        """Sort data by time. This function is defined so that tests will pass."""
        self._isort = np.argsort(self.toas, kind="mergesort")
        self._iisort = np.zeros(len(self._isort), dtype=int)
        for ii, p in enumerate(self._isort):
            self._iisort[p] = ii

    @classmethod
    def read_feather(cls, filename):
        f = feather.read_table(filename)
        self = FeatherPulsar()

        for array in FeatherPulsar.columns:
            if array in f.column_names:
                setattr(self, array, f[array].to_numpy())

        for array in FeatherPulsar.vector_columns:
            cols = [c for c in f.column_names if c.startswith(array)]
            setattr(self, array, np.array([f[col].to_numpy() for col in cols]).swapaxes(0, 1).copy())

        for array in FeatherPulsar.tensor_columns:
            rows = sorted(set(["_".join(c.split("_")[:-1]) for c in f.column_names if c.startswith(array)]))
            cols = [[c for c in f.column_names if c.startswith(row)] for row in rows]
            setattr(
                self,
                array,
                np.array([[f[col].to_numpy() for col in row] for row in cols]).swapaxes(0, 2).swapaxes(1, 2).copy(),
            )

        self.flags = {}
        for array in [c for c in f.column_names if c.startswith("flags_")]:
            self.flags["_".join(array.split("_")[1:])] = f[array].to_numpy().astype("U")

        meta = json.loads(f.schema.metadata[b"json"])
        for attr in FeatherPulsar.metadata:
            if attr in meta:
                setattr(self, attr, meta[attr])
            else:
                print(f"Pulsar.read_feather: cannot find {attr} in feather file {filename}.")

        if "noisedict" in meta:
            setattr(self, "noisedict", meta["noisedict"])

        self.sort_data()

        return self

    def to_list(a):
        return a.tolist() if isinstance(a, np.ndarray) else a

    def save_feather(self, filename, noisedict=None):
        self._toas = self._toas.astype(float)
        pydict = {array: getattr(self, array) for array in FeatherPulsar.columns}

        pydict.update(
            {
                f"{array}_{i}": getattr(self, array)[:, i]
                for array in FeatherPulsar.vector_columns
                for i in range(getattr(self, array).shape[1])
            }
        )

        pydict.update(
            {
                f"{array}_{i}_{j}": getattr(self, array)[:, i, j]
                for array in FeatherPulsar.tensor_columns
                for i in range(getattr(self, array).shape[1])
                for j in range(getattr(self, array).shape[2])
            }
        )

        pydict.update({f"flags_{flag}": self.flags[flag] for flag in self.flags})

        meta = {}
        for attr in Pulsar.metadata:
            if hasattr(self, attr):
                meta[attr] = Pulsar.to_list(getattr(self, attr))
            else:
                print(f"Pulsar.save_feather: cannot find {attr} in Pulsar {self.name}.")

        # use attribute if present
        noisedict = getattr(self, "noisedict", None) if noisedict is None else noisedict
        if noisedict:
            # only keep noisedict entries that are for this pulsar (requires pulsar name to be first part of the key!)
            meta["noisedict"] = {par: val for par, val in noisedict.items() if par.startswith(self.name)}

        feather.write_feather(Table.from_pydict(pydict, metadata={"json": json.dumps(meta)}), filename)


def Pulsar(*args, **kwargs):
    featherfile = [x for x in args if isinstance(x, str) and x.endswith(".feather")]
    if featherfile:
        return FeatherPulsar.read_feather(featherfile[0])
    featherfile = kwargs.get("filepath", None)
    if featherfile:
        return FeatherPulsar.read_feather(featherfile)

    ephem = kwargs.get("ephem", None)
    clk = kwargs.get("clk", None)
    bipm_version = kwargs.get("bipm_version", None)
    planets = kwargs.get("planets", True)
    sort = kwargs.get("sort", True)
    drop_t2pulsar = kwargs.get("drop_t2pulsar", True)
    drop_pintpsr = kwargs.get("drop_pintpsr", True)
    timing_package = kwargs.get("timing_package", None)
    if timing_package is not None:
        timing_package = timing_package.lower()

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

        if timing_package is None:
            if t2 is not None:
                timing_package = "tempo2"
            elif pint is not None:  # pragma: no cover
                timing_package = "pint"
            else:  # pragma: no cover
                raise ValueError("No timing package available with which to load a pulsar")

        # get current directory
        cwd = os.getcwd()
        try:
            # Change directory to the base directory of the tim-file to deal with
            # INCLUDE statements in the tim-file
            os.chdir(dirname)
            if timing_package.lower() == "tempo2":
                if t2 is None:  # pragma: no cover
                    raise ValueError("tempo2 requested but tempo2 is not available")
                # hack to set maxobs
                maxobs = get_maxobs(reltimfile) + 100
                t2pulsar = t2.tempopulsar(relparfile, reltimfile, maxobs=maxobs, ephem=ephem, clk=clk)
                return Tempo2Pulsar(
                    t2pulsar,
                    sort=sort,
                    drop_t2pulsar=drop_t2pulsar,
                    planets=planets,
                    par_name=relparfile,
                    tim_name=reltimfile,
                )
            elif timing_package.lower() == "pint":
                if pint is None:  # pragma: no cover
                    raise ValueError("PINT requested but PINT is not available")
                if (clk is not None) and (bipm_version is None):
                    bipm_version = clk.split("(")[1][:-1]
                model, toas = get_model_and_toas(
                    relparfile, reltimfile, ephem=ephem, bipm_version=bipm_version, planets=planets
                )
                os.chdir(cwd)
                return PintPulsar(toas, model, sort=sort, drop_pintpsr=drop_pintpsr, planets=planets)
            else:
                raise ValueError(f"Unknown timing package {timing_package}")
        finally:
            os.chdir(cwd)
    raise ValueError("Pulsar (par/tim) not specified in {args} or {kwargs}")
