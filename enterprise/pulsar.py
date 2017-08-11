# pulsar.py

# Class containing pulsar data from timing package [tempo2/PINT].

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import enterprise
import numpy as np
from ephem import Ecliptic, Equatorial
import os
import json

try:
    import cPickle as pickle
except:
    import pickle

try:
    import libstempo as t2
except ImportError:
    print('ERROR: Must have libstempo package installed!')
    t2 = None

try:
    import pint.toa as toa
    import pint.models.model_builder as mb
    from pint.models import TimingModel
    from pint.residuals import resids
except ImportError:
    print('No PINT? Meh...')
    pint = None

import astropy.units as u


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
        lines = [ln for ln in flines if not ln.startswith('C')]
        if any(map(lambda x: 'INCLUDE' in x, lines)):
            for line in filter(lambda x: 'INCLUDE' in x, lines):
                maxobs += get_maxobs(line.split()[-1])
        else:
            maxobs = sum(1 for line in lines if line.rstrip('\n'))
    return maxobs


class BasePulsar(object):
    """Abstract Base Class for Pulsar objects."""

    def _get_pdist(self):
        dfile = enterprise.__path__[0] + '/datafiles/pulsar_distances.json'
        with open(dfile, 'r') as fl:
            pdict = json.load(fl)

        try:
            pdist = tuple(pdict[self.name])
        except KeyError:
            # TODO: should use logging here
            msg = 'WARNING: Could not find pulsar distance for '
            msg += 'PSR {0}.'.format(self.name)
            msg += ' Setting value to 1 with 20% uncertainty.'
            print(msg)
            pdist = (1.0, 0.2)
        return pdist

    def _get_radec_from_ecliptic(self, elong, elat):
        # convert via pyephem
        try:
            ec = Ecliptic(elong, elat)

            # check for B name
            if 'B' in self.name:
                epoch = '1950'
            else:
                epoch = '2000'
            eq = Equatorial(ec, epoch=str(epoch))
            raj = np.double(eq.ra)
            decj = np.double(eq.dec)

        # TODO: should use logging here
        except TypeError:
            msg = 'WARNING: Cannot fine sky location coordinates '
            msg += 'for PSR {0}. '.format(self.name)
            msg += 'Setting values to 0.0'
            print(msg)
            raj = 0.0
            decj = 0.0

        return raj, decj

    def _get_pos(self):
        return np.array([np.cos(self._raj) * np.cos(self._decj),
                         np.sin(self._raj) * np.cos(self._decj),
                         np.sin(self._decj)])

    def sort_data(self):
        """Sort data by time."""
        if self._sort:
            self._isort = np.argsort(self._toas, kind='mergesort')
            self._iisort = np.zeros(len(self._isort), dtype=np.int)
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
            mask = np.logical_and(self._toas >= start_time * 86400,
                                  self._toas <= end_time * 86400)

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
        if hasattr(self, 't2pulsar'):
            del self.t2pulsar

        if outdir is None:
            outdir = os.getcwd()

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(outdir + '/{0}.pkl'.format(self.name), 'w') as f:
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
        bflags = ['flag'] * nobs
        check = lambda i, fl: fl in self._flags and self._flags[fl][i] != ''
        flags = [['group'], ['g'], ['sys'], ['i'], ['f'], ['fe', 'be']]
        for ii in range(nobs):
            # TODO: make this cleaner
            for f in flags:
                if np.all(list(map(lambda xx: check(ii, xx), f))):
                    bflags[ii] = '_'.join(self._flags[x][ii] for x in f)
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
    def planetssb(self):
        """Return planetary position vectors at all timestamps"""
        return self._planetssb


class PintPulsar(BasePulsar):

    def __init__(self, toas, model, sort=True, planets=True):

        self._sort = sort
        self.planets = planets
        self.name = model.PSR.value

        self._toas = np.array(toas.table['tdbld'], dtype='float64') * 86400
        self._residuals = np.array(resids(toas, model).time_resids.to(u.s),
                                   dtype='float64')
        self._toaerrs = np.array(toas.get_errors().to(u.s), dtype='float64')
        self._designmatrix = model.designmatrix(toas.table)[0]
        self._ssbfreqs = np.array(model.barycentric_radio_freq(toas.table),
                                  dtype='float64')

        # fitted parameters
        self.fitpars = ['Offset'] + [par for par in model.params
                                     if not getattr(model, par).frozen]

        # set parameters
        spars = [par for par in model.params]
        self.setpars = [sp for sp in spars if sp not in self.fitpars]

        self._flags = {}
        for ii, obsflags in enumerate(toas.get_flags()):
            for jj, flag in enumerate(obsflags):

                if flag not in list(self._flags.keys()):
                    self._flags[flag] = [''] * toas.ntoas

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
        self._planetssb = self._get_planetssb()

        self.sort_data()

    def _get_radec(self, model):
        if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
            return (model.RAJ.value, model.DECJ.value)
        else:
            # TODO: better way of dealing with units
            d2r = np.pi / 180
            elong, elat = model.ELONG.value, model.ELAT.value
            return self._get_radec_from_ecliptic(elong*d2r, elat*d2r)

    def _get_planetssb(self):
        return None


class Tempo2Pulsar(BasePulsar):

    def __init__(self, t2pulsar, sort=True,
                 drop_t2pulsar=True, planets=True):

        self._sort = sort
        self.t2pulsar = t2pulsar
        self.planets = planets
        self.name = str(t2pulsar.name)

        self._toas = np.double(t2pulsar.toas()) * 86400
        self._residuals = np.double(t2pulsar.residuals())
        self._toaerrs = np.double(t2pulsar.toaerrs) * 1e-6
        self._designmatrix = np.double(t2pulsar.designmatrix())
        self._ssbfreqs = np.double(t2pulsar.ssbfreqs()) / 1e6

        # fitted parameters
        self.fitpars = ['Offset'] + list(map(str, t2pulsar.pars()))

        # set parameters
        spars = list(map(str, t2pulsar.pars(which='set')))
        self.setpars = [sp for sp in spars if sp not in self.fitpars]

        self._flags = {}
        for key in t2pulsar.flags():
            self._flags[key] = t2pulsar.flagvals(key)

        self._pdist = self._get_pdist()
        self._raj, self._decj = self._get_radec(t2pulsar)
        self._pos = self._get_pos()
        self._planetssb = self._get_planetssb()

        self.sort_data()

        if drop_t2pulsar:
            del self.t2pulsar

    def _get_radec(self, t2pulsar):
        if 'RAJ' in np.concatenate((t2pulsar.pars(which='fit'),
                                    t2pulsar.pars(which='set'))):
            return (np.double(t2pulsar['RAJ'].val),
                    np.double(t2pulsar['DECJ'].val))

        else:
            # use ecliptic coordinates
            elong = t2pulsar['ELONG'].val
            elat = t2pulsar['ELAT'].val
            return self._get_radec_from_ecliptic(elong, elat)

    def _get_planetssb(self):
        planetssb = None
        if self.planets:
            for ii in range(1, 10):
                tag = 'DMASSPLANET' + str(ii)
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
        return planetssb


def Pulsar(*args, **kwargs):

    ephem = kwargs.get('ephem', None)
    planets = kwargs.get('planets', True)
    sort = kwargs.get('sort', True)
    drop_t2pulsar = kwargs.get('drop_t2pulsar', True)
    timing_package = kwargs.get('timing_package', 'tempo2')

    if pint:
        toas     = list(filter(lambda x: isinstance(x, toa.TOAs), args))
        model    = list(filter(lambda x: isinstance(x, TimingModel), args))

    t2pulsar = list(filter(lambda x: isinstance(x, t2.tempopulsar), args))
    
    parfile  = list(filter(lambda x: isinstance(x, str) and
                           x.split('.')[-1] == 'par', args))
    timfile  = list(filter(lambda x: isinstance(x, str) and
                           x.split('.')[-1] in ['tim', 'toa'], args))

    if pint and toas and model:
        return PintPulsar(toas[0], model[0], sort=sort, planets=planets)
    elif t2pulsar:
        return Tempo2Pulsar(t2pulsar, sort=sort, drop_t2pulsar=drop_t2pulsar,
                            planets=planets)
    elif parfile and timfile:
        # Check whether the two files exist
        if not os.path.isfile(parfile[0]) or not os.path.isfile(timfile[0]):
            msg = 'Cannot find parfile {0} or timfile {1}!'.format(
                parfile[0], timfile[0])
            raise IOError(msg)

        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(timfile[0])
        dirname = timfiletup[0] or './'
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile[0], dirname)

        # get current directory
        cwd = os.getcwd()

        # Change directory to the base directory of the tim-file to deal with
        # INCLUDE statements in the tim-file
        os.chdir(dirname)

        if timing_package.lower() == 'pint':
            if ephem is None:
                ephem = 'DE421'
            toas = toa.get_TOAs(reltimfile, ephem=ephem, planets=planets)
            model = mb.get_model(relparfile)
            os.chdir(cwd)
            return PintPulsar(toas, model, sort=sort, planets=planets)

        elif timing_package.lower() == 'tempo2':

            # hack to set maxobs
            maxobs = get_maxobs(reltimfile)
            t2pulsar = t2.tempopulsar(relparfile, reltimfile,
                                      maxobs=maxobs, ephem=ephem)
            os.chdir(cwd)
            return Tempo2Pulsar(t2pulsar, sort=sort,
                                drop_t2pulsar=drop_t2pulsar,
                                planets=planets)
    else:
        print('Unknown arguments {}'.format(args))
