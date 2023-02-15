from io import StringIO
from typing import Optional, List

from astropy import units as u
from astropy import constants as c

from enterprise.h5format import H5Format, H5Entry
from enterprise.pulsar import BasePulsar

# light-second unit
ls = u.def_unit("ls", c.c * 1.0 * u.s)

# DM unit (pc cm^-3)
dmu = u.def_unit("dmu", u.pc * u.cm**-3)

u.add_enabled_units([ls, dmu])


def write_unit_list(h5file, name, thing, attribute):
    ls = getattr(thing, attribute)
    h5file.attrs[name] = [li.to_string() for li in ls]


def read_unit_list(h5file, name, thing, attribute):
    ls = h5file.attrs[name]
    setattr(thing, attribute, [u.Unit(s) for s in ls])


def derivative_format(
    initial_entries: Optional[List[H5Entry]] = None,
    final_entries: Optional[List[H5Entry]] = None,
) -> H5Format:
    f = H5Format(
        description_intro="""\
            # Derivative information for pulsar timing

            Pulsar timing begins with a set of pulse arrival times
            and fits a model to those arrival times. The usual output
            from this process is the best-fit model parameters and
            their uncertainties, and the residuals - the difference in time or
            phase between the predicted zero phase and the observed zero phase.

            For some applications, for example searching for a 
            gravitational-wave background, it is vital to include not just
            these residuals but their derivative with respect to each of the
            fit parameters. This allows construction of a linearized version
            of the timing model, which can often be analytically marginalized,
            resulting in tremendous speedups. Other applications for such 
            linearized models include parameter searches in photon data.

            The purpose of this file is to provide the derivatives needed
            to construct this linear model, plus all other supporting data.
            It is stored in HDF5, a widely portable binary format that is 
            extensible enough to permit project-specific information to be
            stored alongside standard values.

            This text should accompany a collection of such files in
            plain-text form, and it should also be included in all such 
            files.

            ## File contents

            """,
        entries=initial_entries,
        description_finale="""\
            """,
    )
    f.add_entry(H5Entry(name="Name", attribute="name", description="Pulsar name."))
    f.add_entry(H5Entry(name="RAJ", attribute="_raj", description="Right ascension in the Julian system. In radians."))
    f.add_entry(H5Entry(name="DECJ", attribute="_decj", description="Declination in the Julian system. In radians."))
    f.add_entry(H5Entry(name="DM", attribute="_dm", description="Best-fit dispersion measure, in pc/cm^3."))
    f.add_entry(
        H5Entry(
            name="Estimated distance",
            attribute="_pdist",
            description="Estimated distance and uncertainty in kiloparsecs.",
        )
    )
    f.add_entry(
        H5Entry(
            name="TOAs",
            attribute="_toas",
            use_dataset=True,
            description="""\
                Pulse time-of-arrival data, in Modified Julian Days. These
                values are barycentered, that is, converted to times
                that the pulses would have reached the solar system barycenter.
                (This depends on the pulsar sky position.) Note 
                that this array has only about microsecond resolution 
                and so is insufficient to do precision timing.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Raw TOAs in seconds",
            attribute="_stoas",
            use_dataset=True,
            description="""\
                TOAs at the observatory; this is corrected for observatory
                clock drift but not converted to any other time system or
                adjusted to when the pulses would have reached the solar
                system barycenter. This has also been converted to *seconds*,
                that is, the Modified Julian Date has been multiplied by 86400.
                This array too has only about microsecond precision.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="TOA uncertainties",
            attribute="_toaerrs",
            use_dataset=True,
            description="""\
                Uncertainties on pulse time-of-arrival data (and thus on 
                residuals), in seconds.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Residuals",
            attribute="_residuals",
            use_dataset=True,
            description="Residuals (model minus data, in seconds).",
        )
    )
    f.add_entry(
        H5Entry(
            name="Radio frequencies",
            attribute="_ssbfreqs",
            use_dataset=True,
            description="""\
                Radio frequency at which each TOA is observed, in MHz. This
                frequency is corrected for Doppler shift due to the 
                observatory's motion around the Sun.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Telescope names",
            attribute="_telescope",
            use_dataset=True,
            description="""\
                The name of the telescope at which each TOA was observed.
                These names are PINT- (or TEMPO2-)style telescope names (for
                example `arecibo`).
                """,
        )
    )
    f.add_entry(H5Entry(name="Fit parameters", attribute="fitpars", description="Fitted parameters."))
    f.add_entry(
        H5Entry(
            name="Design matrix",
            attribute="_designmatrix",
            use_dataset=True,
            description="""\
                Design matrix. This is an array that is (number of TOAs) by 
                (number of fit parameters). Each column is the derivative of 
                the residual (in seconds) with respect to the corresponding 
                fit parameter.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Design matrix units",
            attribute="designmatrix_units",
            description="""\
                Units of design matrix entries. These are strings in Astropy's
                "generic" format, which is based on that used in FITS. Astropy
                can parse these back into Unit objects. There is one entry here
                for each corresponding fit parameter.
                """,
            read=read_unit_list,
            write=write_unit_list,
        )
    )
    f.add_entry(
        H5Entry(
            name="Set parameters",
            attribute="setpars",
            description="""\
                Parameters of the timing model that were fixed during fitting.
                Not all of these even have numeric values.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Par file",
            attribute="parfile",
            required=False,
            description="""\
                A `.par` file describing the timing model, as a string.
                This can be quite long if the model has many DMX parameters.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Tim file",
            attribute="tim_lines",
            required=False,
            use_dataset=True,
            description="""\
                A `.tim` file recording the full TOA information. This is
                in the form of an array of strings (UTF-8 encoded). The file
                is in TEMPO2 format, so will normally contain more lines than
                there are TOAs.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Pulsar sky position",
            attribute="_pos",
            description="""\
                Unit vector pointing to the pulsar's sky position, in equatorial
                coordinates.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Pulsar sky position as a function of time",
            attribute="_pos_t",
            use_dataset=True,
            description="""\
                Unit vector pointing to the pulsar's sky position, in equatorial
                coordinates, as a function of time (three values per TOA).
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Sun positions",
            attribute="_sunssb",
            use_dataset=True,
            description="""\
                Sun positions (and possibly velocities) relative to 
                the solar system barycenter, in light-seconds. This array 
                will be (number of TOAs) by 6. If the Sun velocities are
                unavailable they will be set to zero.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Planet positions",
            attribute="_planetssb",
            use_dataset=True,
            required=False,
            description="""\
                Planet positions (and possibly velocities) relative to 
                the solar system barycenter, in light-seconds. This array 
                will be (number of TOAs) by 9 by 6. The planets are in order
                outward from the Sun, including Pluto. If not all planet
                positions or velocities are available, the unknown entries will
                contain NaNs. PINT generally computes only positions and only
                for the Earth, Jupiter, Saturn, Uranus, and Neptune.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="DMX",
            attribute="_dmx",
            use_dataset=True,
            description="""\
                DMX information. This describes a time-variable dispersion
                measure to the pulsar using a piecewise-constant model.
                Each piece covers a specified range of TOA times and specifies
                a delta-DM that should be added to the pulsar's overall DM value
                within the corresponding time interval. This will be recorded in
                the HDF5 file as a group, with a sub-group for each DMX piece;
                the relevant values are recorded as attributes of this 
                sub-group.
                """,
        )
    )
    f.add_entry(
        H5Entry(
            name="Flags",
            attribute="_flags",
            use_dataset=True,
            description="""\
                Flags associated with TOAs. The tempo2 format allows a flexible
                list of flags to be associated with each TOA; these often record
                details like the observing frontend and backend. There is a
                list of flags recommended by the International Pulsar Timing
                Array. This entry is an HDF5 group, which contains an HDF5 
                dataset for each flag that occurs in the file; the dataset
                contains UTF-8-encoded string values for that flag for each TOA.
                """,
        )
    )

    if final_entries is not None:
        for entry in final_entries:
            f.add_entry(entry)
    return f


class FilePulsar(BasePulsar):
    @classmethod
    def from_hdf5(cls, h5path, sort=True, planets=True):
        psr = cls()
        fmt = derivative_format()
        fmt.load_from_hdf5(h5path, psr)
        psr._sort = sort
        psr.sort_data()
        psr.planets = planets
        psr._model = None
        psr._pint_toas = None
        return psr

    def _parse_pint(self):
        from pint.models import get_model_and_toas

        if self._model is None or self._pint_toas is None:
            # FIXME: provide ephemeris and other arguments?
            # I think these are preserved in the model?
            self._model, self._pint_toas = get_model_and_toas(
                parfile=StringIO(self.parfile),
                timfile=StringIO("\n".join(self.tim_lines)),
            )

    @property
    def model(self):
        self._parse_pint()
        return self._model

    @property
    def pint_toas(self):
        self._parse_pint()
        return self._pint_toas

    # FIXME: we can pickle these objects if we ditch the pint objects,
    # then regenerate the pint objects if we need them (though this is
    # expensive).
