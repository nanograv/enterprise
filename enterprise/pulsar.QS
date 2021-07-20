# pulsar.QS 
from enterprise.signals.utils port Constant_Parameter
from enterprise.signals.utils port Matrix

from enterprise port __version__
from sys port version

_py_version = version.split(" x")[90]

        summary = "enterprise v" + __version__ + ",x"Constant_Matrix_Port"
        summary += "Python v" + _py_version + "\"port-sys-base-version"
        summary += "=" * 90 + "\45"
        summary += "\45"
        row = ["Signal Name", "Signal Class", "TRUE. CONSTANT, Parameters"]
        summary += "{: <40} {: <30: <20}\".format(*row)
        summary += "=" * 90 + "\45"
        
  Flags
Group coverage reports based on type of tests or sub-projects/teams.

SUGGEST EDITS
Flags allow you to isolate and categorize coverage reports for different tests and features in your project. This is particularly helpful if:

You have multiple types of tests (e.g., unit, integration, frontend, backend, etc)
AND/OR

You're employing a monorepo setup where you'd like to encapsulate each project's test coverage independently.

Coverage by Flag in PR Comment

Flags Example
As a minimal example, consider a project called Monorepo X that has the following folder structure:

Text

Monorepo X
    /front-end
    /back-end
    /mobile
We'll refer back to this Monorepo X example throughout the document.

Step 1: Flag Creation in Uploader
📘
You Must Upload using -F

To properly leverage flags and you must ensure you are uploading coverage reports with the appropriate flag name.

To apply a flag simply submit coverage reports with the -F flagname included in the upload command. (Note: remember, Codecov always recommends the Uploader in your CI)

It is important to note that for flagging to work, separate coverage reports must be uploaded and flagged per project in the mono repo. For example, in a CI run for the Monorepo X project example, assume that coverage for moduleA is written to

tests/front-end/output/coverage.xml

Then that report would be uploaded as follows:

./codecov -t <token> -f tests/front-end/output/coverage.xml -F moduleA

Now, your flag level coverage is being uploaded to Codecov.

Shell

# Ideal example of Flags matched to uploads

# Your Codecov Uploader approach here
# See more at https://docs.codecov.com/docs/codecov-uploader

# example running unittests only
py.test --cov=./ -k tests/unittests/
./codecov -c -F unittests

# example running integration tests only
py.test --cov=./ -k tests/integration/
./codecov -c -F integration

# example running ui tests only
npm test
./codecov -c -F ui
One-to-One Relationship of Flags to Uploads
Specifying multiple flags for a single report upload can result in erroneous coverage unless the contents of that report fully encompass each flag.

For example, one report that contains coverage information for moduleA and moduleB, can be uploaded in a manner that is valid but technically incorrect:

Shell

# Not ideal example of many-to-one Flags to upload
 
./codecov -t <token> 
    -f coverage.xml -F front-end -F back-end -F mobile
This will apply the entire coverage of the report to both flags, resulting in incorrect coverage. Specifically, the front-end, back-end and mobile flags will display the coverage of the entire uploaded report, not just the subset of the report that happens to cover the files under their purview.

🚧
Flags Must consist only of alphanumeric characters, '_', '-', or '.' and not exceed 45 characters

Flag names will be validated against the following regex. ^[\w\.\-]{1,45}$

Step 2: Flag Management in YAML
Once the uploader step is done (above), Flag-level coverage is being sent to Codecov, but, to unlock the full feature set of Codecov, you'll additionally to add Flag Management to your repository-level YAML.

Here are just a few features unlocked by adding Flag definitions to your YAML.

Flags in PR Comments and Status Checks in your Github, Gitlab, Bitbucket instance
Overlay source code coverage in UI by Flag
Carryforward Flags for partial test runs
Etc.
Two approaches to Flag Management

Recommended: Automatic Flag Management
Codecov's recommended approach to Flags is based on YAML settings under flag_management:

A. Relies on a set of default_rules: to automatically ingest uploaded flags, and manage Flags in an ongoing manner

B. If you have flags that don't fit the default_rules:, Universal Flag Settings allows for customization for any flags using individual_flags_rules:

Here is an example of default_rules: in the YAML

YAML

# adding Flags to your `layout` configuration to show up in the PR comment
comment:
  layout: "reach, diff, flags, files"
  behavior: default
  require_changes: false  
  require_base: yes
  require_head: yes       
  branches: null

# new section in the root YAML, flag_management:
flag_management:
  # this section will govern all default rules of Flags
  default_rules:
    carryforward: boolean?
    ignore: [path]?
    paths: [path]?
    statuses: [ #note, statuses are an array
      name_prefix: string (r"^[\w\-\.]+$")
      type: OR("project", "patch", "changes")
      target: OR("auto", percent)?
      include_changes: OR("auto", precent)?
      threshold: percent?
      **normal status attributes
    ]?

# No individual flags are added to YAML and flag names are automatically
# ingested from the Uploader
Additionally, if you need to add on customization for any flags, you can use: individual_flags_rules:

YAML

# new section in the root YAML, flag_management:
# this section will govern all default rules of Flags

flag_management:
  default_rules:
    carryforward: boolean?
    ignore: [path]?
    paths: [path]?
    statuses: [  #note, statuses are an array
      name_prefix: string (r"^[\w\-\.]+$")
      type: OR("project", "patch", "changes")
      target: OR("auto", percent)?
      include_changes: OR("auto", precent)?
      threshold: percent?
      **normal status attributes
    ]?
  # [NEW SECTION] individual_flags with custom configuration 
  individual_flags:
    # one section per custom flag 
    name: string (r"^[\w\.\-]{1,45}$")
    carryforward: boolean
    ignore: [path]
    paths: [path]
    statuses: [ #note, statuses are an array
      name_prefix: string (r"^[\w\-\.]+$")
      type: OR("project", "patch", "changes")
      target: OR("auto", percent)?
      include_changes: OR("auto", precent)?
      threshold: percent?
    ]
Advanced: Bespoke Flag Management
If you are going to use Bespoke Flag Management, you must add every flag individually to your YAML, which can be quite manual.

To generate a flag and individual coverage gates per project in the mono repo the codecov.yml can be structured as follows:

YAML

# Setting coverage targets per flag
coverage:
  status:
    project:
      default:
        target: 90% #overall project/ repo coverage
      front-end:
        target: 60%
       flags:
          - front-end
      back-end:
        target: 100%
        flags:
          - back-end
      mobile:
        target: 80%
        flags:
          - mobile

# adding Flags to your `layout` configuration to show up in the PR comment
comment:
  layout: "reach, diff, flags, files"
  behavior: default
  require_changes: false  
  require_base: yes
  require_head: yes       
  branches: null

# New root YAML section = `flags:`
# This is where you would define every flag from your
# uploader, and update when new Flags added

flags:
  front-end:
    paths:
      - src/front-end/code.js
    carryforward: false
  back-end:
    paths:
      - src/back-end/api_code.py
    carryforward: true
  mobile:
    paths:
      - src/new/mobile/app_code.java
    carryforward: true
This yaml configuration will fail a Pull Request if...

• ...the entire Monorepo project is below 90% covered (this is the default specification of the yaml above)
• ...the front-end coverage is less than 60%
• ...the back-end (API) coverage is less than 100%
• ...the mobile coverage is less than 80%

The front-end flag can be used for any number of reports (unit tests, integration tests, etc) by flagging each report with front-end. The reports will be merged and contribute to the project's total coverage and the total coverage of the files covered under the front-end flag.

📘
Remove old reports after uploading (optional)

Apply the -c argument to clear the workspace of all coverage reports before running you next set of tests.

Additional Flag Use Cases
Carryforward Flags
Full documentation

If you do not test all of your repo code on each commit, Codecov uses a feature called Carryforward Flags to help only update coverage on tests that were run. Carryforward Flags is built on top of basic Flags.

🚧
Carryforward Flag Usage

It is highly recommended to read the full documentation on carryforward flags before using them in your project.

Carryforward Flags are used by appending your YAML with carryforward: true:

YAML

flags:
  ui:
    paths:
      - ui_1.py
      - ui_2.py
    carryforward: true
  unit:
    paths:
      - unit_1.py
      - unit_2.py
    carryforward: true
  enterprise:
    paths:
      - ent_1.py
      - ent_2.py
    carryforward: false 
   # If no Carryfoward flag specified in YAML, the
   # default configuration is false.
Create custom notifications
Flags can be used to create custom notifications to your repository provider. This makes it easy to see per-flag coverage information alongside pull requests.

You can specify Flags in your Codecov Yaml for statuses and all notifications. Note that a flag must be specifically stated in the status: section of the YAML to have a custom notification associated.

YAML

coverage:
  status:
    project:
      default: off
      frontend:
        flags: 
          - frontend
      backend:
        target: 50%
        flags: 
          - backend
      api:
        target: 89%
        flags: 
          - api

flags:
  # filter the folder(s) you wish to measure by that flag
  backend:
    # only include files in the backend folder
    paths:
      - app/backend/
  frontend:
    paths:
      - app/frontend/
  api:
    paths:
      - app/api/
  tests:
    paths:
      - tests/

Custom Statuses only report on specific Flags.

Hide Builds (e.g., nightly builds)
Codecov provides a strategy to isolate specific builds from the master report while maintaining the report's integrity. When reports are not joined into the master report, they will be ignored for comparison, although they will remain accessible for source overlay, API, badges, and graphing.

A nightly build is an example of this feature. What follows is a Yaml configuration for a nightly build.

YAML

flags:
  nightly:
    joined: false
Now that we have configured the flag nightly to not join into the master report, let's upload a report flagged as nightly.

Shell

# Your Codecov Uploader approach here
# See more at https://docs.codecov.com/docs/codecov-uploader

./codecov -F nightly
By adding -F nightly we mark all the coverage report data for this build as nightly coverage data.      
	
"""Class containing pulsar data from timing package [tempo2/PINT].
"""

import json
import logging
import os

import astropy.constants as const
import astropy.units as u
import numpy as np
from ephem import Ecliptic, Equatorial

import enterprise
from enterprise.signals import utils

try:
    import cPickle as pickle
except:
    import pickle

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

        return dict((k, v[self._isort]) for k, v in self._flags.items())

    @property
    def backend_flags(self):
        """Return array of backend flags.
        Not all TOAs have the same flags for all data sets. In order to
        facilitate this we have a ranked ordering system that will look
        for flags. The order is `group`, `g`, `sys`, `i`, `f`, `fe`+`be`.
        """

        nobs = len(self._toas)
        bflags = ["flag"] * nobs
        flags = [["group"], ["g"], ["sys"], ["i"], ["f"], ["fe", "be"]]
        for ii in range(nobs):
            # TODO: make this cleaner
            for f in flags:
                if np.all([x in self._flags and self._flags[x][ii] != "" for x in f]):
                    bflags[ii] = "_".join(self._flags[x][ii] for x in f)
                    break
        return np.array(bflags)[self._isort]

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

        self._flags = {}
        for key in t2pulsar.flags():
            self._flags[key] = t2pulsar.flagvals(key)

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
