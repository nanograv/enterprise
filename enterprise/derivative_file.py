from typing import Optional, List
from enterprise.h5format import H5Format, H5Entry


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

            """,
        entries=initial_entries,
        description_finale="""\
            """,
    )
    f.add_entry(H5Entry(name="name", attribute="_name", description="Pulsar name."))
    f.add_entry(H5Entry(name="RAJ", attribute="_raj", description="Right ascension in the Julian system. In radians?"))
    f.add_entry(H5Entry(name="DECJ", attribute="_decj", description="Declination in the Julian system. In radians?"))
    f.add_entry(H5Entry(name="DM", attribute="_dm", description="Best-fit dispersion measure, in pc/cm^3."))
    f.add_entry(
        H5Entry(
            name="Estimated distance",
            attribute="_pdist",
            description="Estimated distance and uncertainty (?) in kiloparsecs.",
        )
    )
    f.add_entry(
        H5Entry(
            name="TOAs",
            attribute="_toas",
            description="""\
                Pulse time-of-arrival data, in Modified Julian Days. Note 
                that this column has only about microsecond resolution 
                and so is insufficient to do precision timing.
                """,
        )
    )
    f.add_entry(
        H5Entry(name="Residuals", attribute="_residuals", description="Residuals (model minus data, in seconds).")
    )
    f.add_entry(H5Entry(name="Fit parameters", attribute="fitpars", description="Fitted parameters."))
    f.add_entry(
        H5Entry(
            name="Design matrix",
            attribute="_designmatrix",
            description="""\
                Design matrix. This is an array that is (number of TOAs) by 
                (number of fit parameters). Each column is the derivative of 
                the residual (in seconds?) with respect to the corresponding 
                fit parameter.
                """,
        )
    )
