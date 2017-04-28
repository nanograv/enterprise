# pulsar.py

# Class containing pulsar data from timing package [tempo2/PINT].

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import enterprise
import numpy as np
import tempfile
from ephem import Ecliptic, Equatorial
import os
import json
from collections import OrderedDict

# NOTE: PINT interface is not yet available so just import libstempo
try:
    import libstempo as t2
except ImportError:
    print('ERROR: Must have libstempo package installed!')
    t2 = None


class Pulsar(object):

    def __init__(self, parfile, timfile, maxobs=30000, ephem=None,
                 planets=True):
        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            msg = 'Cannot find parfile {0} or timfile {1}!'.format(
                parfile, timfile)
            raise IOError(msg)

        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(timfile)
        dirname = timfiletup[0]
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile, dirname)

        # get current directory
        cwd = os.getcwd()

        # Change directory to the base directory of the tim-file to deal with
        # INCLUDE statements in the tim-file
        os.chdir(dirname)

        # Load pulsar data from the libstempo library
        # TODO: make sure we specify libstempo>=2.3.1
        self.t2pulsar = t2.tempopulsar(relparfile, reltimfile,
                                       maxobs=maxobs, ephem=ephem)

        # Load the entire par-file into memory, so that we can save it
        with open(relparfile, 'r') as content_file:
            self.parfile_content = content_file.read()

        # Save the tim-file to a temporary file (so that we don't have to deal
        # with 'include' statements in the tim-file), and load that tim-file
        # in memory for storage
        tempfilename = tempfile.mktemp()
        self.t2pulsar.savetim(tempfilename)
        with open(tempfilename, 'r') as content_file:
            self.timfile_content = content_file.read()
        os.remove(tempfilename)

        # Change directory back to where we were
        os.chdir(cwd)

        # get pulsar name
        self.name = str(self.t2pulsar.name)

        # get some private attributes
        self._toas = np.double(self.t2pulsar.toas()) * 86400
        self._residuals = np.double(self.t2pulsar.residuals())
        self._toaerrs = np.double(self.t2pulsar.toaerrs) * 1e-6
        self._designmatrix = np.double(self.t2pulsar.designmatrix())
        self._ssbfreqs = np.double(self.t2pulsar.ssbfreqs()) / 1e6

        # get pulsar distances and uncertainties
        # TODO: use fancier datapath representation
        dfile = enterprise.__path__[0] + '/datafiles/pulsar_distances.json'
        with open(dfile, 'r') as fl:
            pdict = json.load(fl)

        try:
            self._pdist = tuple(pdict[self.name])
        except KeyError:
            # TODO: should use logging here
            msg = 'WARNING: Could not find pulsar distance for '
            msg += 'PSR {0}.'.format(self.name)
            msg += ' Setting value to 1 with 20% uncertainty.'
            print(msg)
            self._pdist = (1.0, 0.2)

        # get par-file parameters
        self.fitpars = ['Offset'] + list(map(str, self.t2pulsar.pars()))

        # make dictionary with tuple of (val, err) for fitted parameters

        # set values for offset
        # TODO: should probably try to set uncertainty here in some way
        self.fitvals = OrderedDict({})
        self.fitvals['Offset'] = (0.0, 0.0)

        # get other fit values
        for par in self.fitpars[1:]:
            self.fitvals[par] = (self.t2pulsar[par].val,
                                 self.t2pulsar[par].err)

        # get set values in par file
        spars = list(map(str, self.t2pulsar.pars(which='set')))
        self.setpars = [sp for sp in spars if sp not in self.fitpars]

        # make dictionary with set parameter values
        self.setvals = OrderedDict({})
        for par in self.setpars:
            self.setvals[par] = self.t2pulsar[par].val

        # get flag dictionary
        # TODO: Deal with heirarchical setting of flags in Models?
        # For example, some IPTA data has group flags but NANOGrav doesn't
        self._flags = {}
        for key in self.t2pulsar.flags():
            self._flags[key] = self.t2pulsar.flagvals(key)

        # get sky location
        if 'RAJ' in np.concatenate((self.fitpars, self.setpars)):
            self._raj = np.double(self.t2pulsar['RAJ'].val)
            self._decj = np.double(self.t2pulsar['DECJ'].val)

        else:
            # use ecliptic coordinates
            elong = self.t2pulsar['ELONG'].val
            elat = self.t2pulsar['ELAT'].val

            # convert via pyephem
            try:
                ec = Ecliptic(elong, elat)

                # check for B name
                if 'B' in self.name:
                    epoch = '1950'
                else:
                    epoch = '2000'
                eq = Equatorial(ec, epoch=str(epoch))
                self._raj = np.double(eq.ra)
                self._decj = np.double(eq.dec)

            # TODO: should use logging here
            except TypeError:
                msg = 'WARNING: Cannot fine sky location coordinates '
                msg += 'for PSR {0}. '.format(self.name)
                msg += 'Setting values to 0.0'
                print(msg)
                self._raj = 0.0
                self._decj = 0.0

        # Get the position vectors of the planets
        self._planetssb = None
        if planets:
            for ii in range(1, 10):
                tag = 'DMASSPLANET' + str(ii)
                self.t2pulsar[tag].val = 0.0
            self.t2pulsar.formbats()
            self._planetssb = np.zeros((len(self._toas), 9, 6))
            self._planetssb[:, 0, :] = self.t2pulsar.mercury_ssb
            self._planetssb[:, 1, :] = self.t2pulsar.venus_ssb
            self._planetssb[:, 2, :] = self.t2pulsar.earth_ssb
            self._planetssb[:, 3, :] = self.t2pulsar.mars_ssb
            self._planetssb[:, 4, :] = self.t2pulsar.jupiter_ssb
            self._planetssb[:, 5, :] = self.t2pulsar.saturn_ssb
            self._planetssb[:, 6, :] = self.t2pulsar.uranus_ssb
            self._planetssb[:, 7, :] = self.t2pulsar.neptune_ssb
            self._planetssb[:, 8, :] = self.t2pulsar.pluto_ssb

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

    def to_pickle(self, savedir):
        """Save object to pickle file."""

        # may have difficulties with this since tempopulsar objects
        # are note pickleable.
        pass

    @property
    def toas(self):
        """Return array of TOAs in seconds."""
        return self._toas

    @property
    def residuals(self):
        """Return array of residuals in seconds."""
        return self._residuals

    @property
    def toaerrs(self):
        """Return array of TOA errors in seconds."""
        return self._toaerrs

    @property
    def freqs(self):
        """Return array of radio frequencies in MHz."""
        return self._ssbfreqs

    @property
    def Mmat(self):
        """Return ntoa x npar design matrix."""
        return self._designmatrix

    @property
    def pdist(self):
        """Return tuple of pulsar distance and uncertainty in kpc."""
        return self._pdist

    @property
    def flags(self):
        """Return a dictionary of tim-file flags."""
        return self._flags

    @property
    def backend_flags(self):
        """Return array of backend flags.

        Not all TOAs have the same flags for all data sets. In order to
        facilitate this we have a ranked ordering system that will look
        for flags. The order is `group`, `sys`, `i`, `f`, `fe`+`be`.

        """

        nobs = len(self._toas)
        bflags = ['flag'] * nobs
        check = lambda i, fl: fl in self._flags and self._flags[fl][i] != ''
        flags = [['group'], ['sys'], ['i'], ['f'], ['fe', 'be']]
        for ii in range(nobs):
            # TODO: make this cleaner
            for f in flags:
                if np.all(map(lambda xx: check(ii, xx), f)):
                    bflags[ii] = '_'.join(self._flags[x][ii] for x in f)
                    break
        return np.array(bflags)

    @property
    def theta(self):
        """Return polar angle of pulsar in radians."""
        return np.pi / 2 - self._decj

    @property
    def phi(self):
        """Return azimuthal angle of pulsar in radians."""
        return self._raj

    @property
    def planetssb(self):
        """Return planetary position vectors at all timestamps"""
        return self._planetssb
