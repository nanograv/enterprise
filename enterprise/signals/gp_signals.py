# gp_signals.py
"""Contains class factories for Gaussian Process (GP) signals.
GP signals are defined as the class of signals that have a basis
function matrix and basis prior vector..
"""

import functools
import itertools
import logging

import numpy as np
import scipy.sparse as sps
from sksparse.cholmod import cholesky

from enterprise.signals import parameter, selections, signal_base, utils
from enterprise.signals.parameter import function
from enterprise.signals.selections import Selection
from enterprise.signals.utils import KernelMatrix

# logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def BasisGP(
    priorFunction,
    basisFunction,
    coefficients=False,
    combine=True,
    selection=Selection(selections.no_selection),
    name="",
):
    """Class factory for generic GPs with a basis matrix."""

    class BasisGP(signal_base.Signal):
        signal_type = "basis"
        signal_name = name
        signal_id = name

        basis_combine = combine

        def __init__(self, psr):
            super(BasisGP, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id
            self._do_selection(psr, priorFunction, basisFunction, coefficients, selection)

        def _do_selection(self, psr, priorfn, basisfn, coefficients, selection):
            sel = selection(psr)

            self._keys = sorted(sel.masks.keys())
            self._masks = [sel.masks[key] for key in self._keys]
            self._prior, self._bases = {}, {}
            self._params, self._coefficients = {}, {}

            for key, mask in zip(self._keys, self._masks):
                pnames = [psr.name, name, key]
                pname = "_".join([n for n in pnames if n])

                self._prior[key] = priorfn(pname, psr=psr)
                self._bases[key] = basisfn(pname, psr=psr)

                for par in itertools.chain(self._prior[key]._params.values(), self._bases[key]._params.values()):
                    self._params[par.name] = par

            if coefficients:
                # we can only create GPCoefficients parameters if the basis
                # can be constructed with default arguments
                # (and does not change size)
                self._construct_basis()

                for key in self._keys:
                    pname = "_".join([n for n in [psr.name, name, key] if n])

                    chain = itertools.chain(self._prior[key]._params.values(), self._bases[key]._params.values())
                    priorargs = {par.name: self._params[par.name] for par in chain}

                    logprior = parameter.Function(functools.partial(self._get_coefficient_logprior, key), **priorargs)

                    size = self._slices[key].stop - self._slices[key].start

                    cpar = parameter.GPCoefficients(logprior=logprior, size=size)(pname + "_coefficients")

                    self._coefficients[key] = cpar
                    self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            ret = []
            for basis in self._bases.values():
                ret.extend([pp.name for pp in basis.params])
            return ret

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", limit=1)
        def _construct_basis(self, params={}):
            basis, self._labels = {}, {}
            for key, mask in zip(self._keys, self._masks):
                basis[key], self._labels[key] = self._bases[key](params=params, mask=mask)

            nc = sum(F.shape[1] for F in basis.values())
            self._basis = np.zeros((len(self._masks[0]), nc))

            # TODO: should this be defined here? it will cache phi
            self._phi = KernelMatrix(nc)

            self._slices = {}
            nctot = 0
            for key, mask in zip(self._keys, self._masks):
                Fmat = basis[key]
                nn = Fmat.shape[1]
                self._basis[mask, nctot : nn + nctot] = Fmat
                self._slices.update({key: slice(nctot, nn + nctot)})
                nctot += nn

        # this class does different things (and gets different method
        # definitions) if the user wants it to model GP coefficients
        # (e.g., for a hierarchical likelihood) or if they do not
        if coefficients:

            def _get_coefficient_logprior(self, key, c, **params):
                self._construct_basis(params)

                phi = self._prior[key](self._labels[key], params=params)

                if phi.ndim == 1:
                    return -0.5 * np.sum(c * c / phi) - 0.5 * np.sum(np.log(phi)) - 0.5 * len(phi) * np.log(2 * np.pi)
                    # note: (2*pi)^(n/2) is not in signal_base likelihood
                else:
                    # TO DO: this code could be embedded in KernelMatrix
                    phiinv, logdet = KernelMatrix(phi).inv(logdet=True)
                    return -0.5 * np.dot(c, np.dot(phiinv, c)) - 0.5 * logdet - 0.5 * phi.shape[0] * np.log(2 * np.pi)

            # MV: could assign this to a data member at initialization
            @property
            def delay_params(self):
                return [pp.name for pp in self.params if "_coefficients" in pp.name]

            @signal_base.cache_call(["basis_params", "delay_params"])
            def get_delay(self, params={}):
                self._construct_basis(params)

                c = np.zeros(self._basis.shape[1])
                for key, slc in self._slices.items():
                    p = self._coefficients[key]
                    c[slc] = params[p.name] if p.name in params else p.value

                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phiinv(self, params):
                return None

        else:

            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)

                for key, slc in self._slices.items():
                    phislc = self._prior[key](self._labels[key], params=params)
                    self._phi = self._phi.set(phislc, slc)
                return self._phi

            def get_phiinv(self, params):
                return self.get_phi(params).inv()

    return BasisGP


def FourierBasisGP(
    spectrum,
    coefficients=False,
    combine=True,
    components=20,
    selection=Selection(selections.no_selection),
    Tspan=None,
    modes=None,
    name="red_noise",
    pshift=False,
    pseed=None,
):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, pshift=pshift, pseed=pseed)
    BaseClass = BasisGP(spectrum, basis, coefficients=coefficients, combine=combine, selection=selection, name=name)

    class FourierBasisGP(BaseClass):
        signal_type = "basis"
        signal_name = "red noise"
        signal_id = name

    return FourierBasisGP


def TimingModel(coefficients=False, name="linear_timing_model", use_svd=False, normed=True):
    """Class factory for marginalized linear timing model signals."""

    if normed is True:
        basis = utils.normed_tm_basis()
    elif isinstance(normed, np.ndarray):
        basis = utils.normed_tm_basis(norm=normed)
    elif use_svd is True:
        if normed is not True:
            msg = "use_svd == True is incompatible with normed != True"
            raise ValueError(msg)
        basis = utils.svd_tm_basis()
    else:
        basis = utils.unnormed_tm_basis()

    prior = utils.tm_prior()
    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

        if coefficients:

            def _get_coefficient_logprior(self, key, c, **params):
                # MV: probably better to avoid this altogether
                #     than to use 1e40 as in get_phi
                return 0

    return TimingModel


@function
def ecorr_basis_prior(weights, log10_ecorr=-8):
    """Returns the ecorr prior.
    :param weights: A vector or weights for the ecorr prior.
    """
    return weights * 10 ** (2 * log10_ecorr)


def EcorrBasisModel(
    log10_ecorr=parameter.Uniform(-10, -5),
    coefficients=False,
    selection=Selection(selections.no_selection),
    name="basis_ecorr",
):
    """Convenience function to return a BasisGP class with a
    quantized ECORR basis."""

    basis = utils.create_quantization_matrix()
    prior = ecorr_basis_prior(log10_ecorr=log10_ecorr)
    BaseClass = BasisGP(prior, basis, coefficients=coefficients, selection=selection, name=name)

    class EcorrBasisModel(BaseClass):
        signal_type = "basis"
        signal_name = "basis ecorr"
        signal_id = name

    return EcorrBasisModel


def BasisCommonGP(priorFunction, basisFunction, orfFunction, coefficients=False, combine=True, name=""):
    class BasisCommonGP(signal_base.CommonSignal):
        signal_type = "common basis"
        signal_name = "common"
        signal_id = name

        basis_combine = combine

        _orf = orfFunction(name)
        _prior = priorFunction(name)

        def __init__(self, psr):
            super(BasisCommonGP, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id

            pname = "_".join([psr.name, name])
            self._bases = basisFunction(pname, psr=psr)

            self._params, self._coefficients = {}, {}

            for par in itertools.chain(
                self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
            ):
                self._params[par.name] = par

            self._psrpos = psr.pos

            if coefficients:
                self._construct_basis()

                # if we're given an instantiated coefficient vector
                # that's what we will use
                if isinstance(coefficients, parameter.Parameter):
                    self._coefficients[""] = coefficients
                    self._params[coefficients.name] = coefficients

                    return

                chain = itertools.chain(
                    self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
                )
                priorargs = {par.name: self._params[par.name] for par in chain}

                logprior = parameter.Function(self._get_coefficient_logprior, **priorargs)

                size = self._basis.shape[1]

                cpar = parameter.GPCoefficients(logprior=logprior, size=size)(pname + "_coefficients")

                self._coefficients[""] = cpar
                self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            return [pp.name for pp in self._bases.params]

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", limit=1)
        def _construct_basis(self, params={}):
            self._basis, self._labels = self._bases(params=params)

        if coefficients:

            def _get_coefficient_logprior(self, c, **params):
                # MV: for correlated GPs, the prior needs to use
                #     the coefficients for all GPs together;
                #     this may require parameter groups

                raise NotImplementedError("Need to implement common prior " + "for BasisCommonGP coefficients")

            @property
            def delay_params(self):
                return [pp.name for pp in self.params if "_coefficients" in pp.name]

            @signal_base.cache_call(["basis_params", "delay_params"])
            def get_delay(self, params={}):
                self._construct_basis(params)

                p = self._coefficients[""]
                c = params[p.name] if p.name in params else p.value
                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phicross(cls, signal1, signal2, params):
                return None

            def get_phiinv(self, params):
                return None

        else:

            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)

                prior = BasisCommonGP._prior(self._labels, params=params)
                orf = BasisCommonGP._orf(self._psrpos, self._psrpos, params=params)

                return prior * orf

            @classmethod
            def get_phicross(cls, signal1, signal2, params):
                prior = BasisCommonGP._prior(signal1._labels, params=params)
                orf = BasisCommonGP._orf(signal1._psrpos, signal2._psrpos, params=params)

                return prior * orf

    return BasisCommonGP


def FourierBasisCommonGP(
    spectrum,
    orf,
    coefficients=False,
    combine=True,
    components=20,
    Tspan=None,
    modes=None,
    name="common_fourier",
    pshift=False,
    pseed=None,
):

    if coefficients and Tspan is None:
        raise ValueError(
            "With coefficients=True, FourierBasisCommonGP " + "requires that you specify Tspan explicitly."
        )

    basis = utils.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, pshift=pshift, pseed=pseed)
    BaseClass = BasisCommonGP(spectrum, basis, orf, coefficients=coefficients, combine=combine, name=name)

    class FourierBasisCommonGP(BaseClass):
        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP._Tmin.append(psr.toas.min())
                FourierBasisCommonGP._Tmax.append(psr.toas.max())

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", 1)
        def _construct_basis(self, params={}):
            span = Tspan if Tspan is not None else max(FourierBasisCommonGP._Tmax) - min(FourierBasisCommonGP._Tmin)
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP


# for simplicity, we currently do not handle Tspan automatically
def FourierBasisCommonGP_ephem(spectrum, components, Tspan, name="ephem_gp"):
    basis = utils.createfourierdesignmatrix_ephem(nmodes=components, Tspan=Tspan)
    orf = utils.monopole_orf()

    return BasisCommonGP(spectrum, basis, orf, name=name)


def FourierBasisCommonGP_physicalephem(
    frame_drift_rate=1e-9,
    d_jupiter_mass=1.54976690e-11,
    d_saturn_mass=8.17306184e-12,
    d_uranus_mass=5.71923361e-11,
    d_neptune_mass=7.96103855e-11,
    jup_orb_elements=0.05,
    sat_orb_elements=0.5,
    model="setIII",
    coefficients=False,
    name="phys_ephem_gp",
):
    """
    Class factory for physical ephemeris corrections as a common GP.
    Individual perturbations can be excluded by setting the corresponding
    prior sigma to None.

    :param frame_drift_rate: Gaussian sigma for frame drift rate
    :param d_jupiter_mass:   Gaussian sigma for Jupiter mass perturbation
    :param d_saturn_mass:    Gaussian sigma for Saturn mass perturbation
    :param d_uranus_mass:    Gaussian sigma for Uranus mass perturbation
    :param d_neptune_mass:   Gaussian sigma for Neptune mass perturbation
    :param jup_orb_elements: Gaussian sigma for Jupiter orbital elem. perturb.
    :param sat_orb_elements: Gaussian sigma for Saturn orbital elem. perturb.
    :param model:            vector basis used by Jupiter and Saturn perturb.;
                             see PhysicalEphemerisSignal, defaults to "setIII"
    :param coefficients:     if True, treat GP coefficients as enterprise
                             parameters; if False, marginalize over them

    :return: BasisCommonGP representing ephemeris perturbations
    """

    basis = utils.createfourierdesignmatrix_physicalephem(
        frame_drift_rate=frame_drift_rate,
        d_jupiter_mass=d_jupiter_mass,
        d_saturn_mass=d_saturn_mass,
        d_uranus_mass=d_uranus_mass,
        d_neptune_mass=d_neptune_mass,
        jup_orb_elements=jup_orb_elements,
        sat_orb_elements=sat_orb_elements,
        model=model,
    )

    spectrum = utils.physicalephem_spectrum()
    orf = utils.monopole_orf()

    return BasisCommonGP(spectrum, basis, orf, coefficients=coefficients, name=name)


def WidebandTimingModel(
    dmefac=parameter.Uniform(pmin=0.1, pmax=10.0),
    log10_dmequad=parameter.Uniform(pmin=-7.0, pmax=0.0),
    dmjump=parameter.Uniform(pmin=-0.01, pmax=0.01),
    dmefac_selection=Selection(selections.no_selection),
    log10_dmequad_selection=Selection(selections.no_selection),
    dmjump_selection=Selection(selections.no_selection),
    dmjump_ref=None,
    name="wideband_timing_model",
):
    """Class factory for marginalized linear timing model signals
    that take wideband TOAs and DMs.  Currently assumes DMX for DM model."""

    basis = utils.unnormed_tm_basis()  # will need to normalize phi otherwise
    prior = utils.tm_prior()  # standard
    BaseClass = BasisGP(prior, basis, coefficients=False, name=name)

    class WidebandTimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "wideband timing model"
        signal_id = name

        basis_combine = False  # should never need to be True

        def __init__(self, psr):
            super(WidebandTimingModel, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id

            # make selection for DMEFACs
            dmefac_select = dmefac_selection(psr)
            self._dmefac_keys = list(sorted(dmefac_select.masks.keys()))
            self._dmefac_masks = [dmefac_select.masks[key] for key in self._dmefac_keys]

            # make selection for DMEQUADs
            log10_dmequad_select = log10_dmequad_selection(psr)
            self._log10_dmequad_keys = list(sorted(log10_dmequad_select.masks.keys()))
            self._log10_dmequad_masks = [log10_dmequad_select.masks[key] for key in self._log10_dmequad_keys]

            # make selection for DMJUMPs
            dmjump_select = dmjump_selection(psr)
            self._dmjump_keys = list(sorted(dmjump_select.masks.keys()))
            self._dmjump_masks = [dmjump_select.masks[key] for key in self._dmjump_keys]

            if self._dmjump_keys == [""] and dmjump is not None:
                raise ValueError("WidebandTimingModel: can only do DMJUMP with more than one selection.")

            # collect parameters

            self._params = {}

            self._dmefacs = []
            for key in self._dmefac_keys:
                pname = "_".join([n for n in [psr.name, key, "dmefac"] if n])
                param = dmefac(pname)

                self._dmefacs.append(param)
                self._params[param.name] = param

            self._log10_dmequads = []
            for key in self._log10_dmequad_keys:
                pname = "_".join([n for n in [psr.name, key, "log10_dmequad"] if n])
                param = log10_dmequad(pname)

                self._log10_dmequads.append(param)
                self._params[param.name] = param

            self._dmjumps = []
            if dmjump is not None:
                for key in self._dmjump_keys:
                    pname = "_".join([n for n in [psr.name, key, "dmjump"] if n])
                    if dmjump_ref is not None:
                        if pname == psr.name + "_" + dmjump_ref + "_dmjump":
                            fixed_dmjump = parameter.Constant(val=0.0)
                            param = fixed_dmjump(pname)
                        else:
                            param = dmjump(pname)
                    else:
                        param = dmjump(pname)

                    self._dmjumps.append(param)
                    self._params[param.name] = param

            # copy psr quantities

            self._ntoas = len(psr.toas)
            self._npars = len(psr.fitpars)

            self._freqs = psr.freqs

            # collect DMX information (will be used to make phi and delay)

            self._dmpar = psr.dm
            self._dm = np.array(psr.flags["pp_dm"], "d")
            self._dmerr = np.array(psr.flags["pp_dme"], "d")

            check = np.zeros_like(psr.toas, "i")

            # assign TOAs to DMX bins

            self._dmx, self._dmindex, self._dmwhich = [], [], []
            for index, key in enumerate(sorted(psr.dmx)):
                dmx = psr.dmx[key]

                if not dmx["fit"]:
                    raise ValueError("WidebandTimingModel: all DMX parameters must be estimated.")

                self._dmx.append(dmx["DMX"])
                self._dmindex.append(psr.fitpars.index(key))
                self._dmwhich.append((dmx["DMXR1"] <= psr.stoas / 86400) & (psr.stoas / 86400 < dmx["DMXR2"]))

                check += self._dmwhich[-1]

            if np.sum(check) != self._ntoas:
                raise ValueError("WidebandTimingModel: cannot account for all TOAs in DMX intervals.")

            if "DM" in psr.fitpars:
                raise ValueError("WidebandTimingModel: DM must not be estimated.")

            self._ndmx = len(self._dmx)

        @property
        def delay_params(self):
            # cache parameters are all DMEFACS, DMEQUADS, and DMJUMPS
            return (
                [p.name for p in self._dmefacs]
                + [p.name for p in self._log10_dmequads]
                + [p.name for p in self._dmjumps]
            )

        @signal_base.cache_call(["delay_params"])
        def get_phi(self, params):
            """Return wideband timing-model prior."""

            # get DMEFAC- and DMEQUAD-adjusted DMX errors
            dme = self.get_dme(params)

            # initialize the timing-model "infinite" prior
            phi = KernelMatrix(1e40 * np.ones(self._npars, "d"))

            # fill the DMX slots with weighted errors
            for index, which in zip(self._dmindex, self._dmwhich):
                phi.set(1.0 / np.sum(1.0 / dme[which] ** 2), index)

            return phi

        def get_phiinv(self, params):
            """Return inverse prior (using KernelMatrix inv)."""
            return self.get_phi(params).inv()

        @signal_base.cache_call(["delay_params"])
        def get_delay(self, params):
            """Return the weighted-mean DM correction that applies for each residual.
            (Will be the same across each DM bin, before measurement-frequency weighting.)"""

            dm_delay = np.zeros(self._ntoas, "d")

            avg_dm = self.get_mean_dm(params)

            for dmx, which in zip(self._dmx, self._dmwhich):
                dm_delay[which] = avg_dm[which] - (self._dmpar + dmx)

            return dm_delay / (2.41e-4 * self._freqs ** 2)

        @signal_base.cache_call(["delay_params"])
        def get_dm(self, params):
            """Return DMJUMP-adjusted DM measurements."""

            return (
                sum(
                    (params[jump.name] if jump.name in params else jump.value) * mask
                    for jump, mask in zip(self._dmjumps, self._dmjump_masks)
                )
                + self._dm
            )

        @signal_base.cache_call(["delay_params"])
        def get_dme(self, params):
            """Return EFAC- and EQUAD-weighted DM errors."""

            return (
                sum(
                    (params[efac.name] if efac.name in params else efac.value) * mask
                    for efac, mask in zip(self._dmefacs, self._dmefac_masks)
                )
                ** 2
                * self._dmerr ** 2
                + (
                    10
                    ** sum(
                        (params[equad.name] if equad.name in params else equad.value) * mask
                        for equad, mask in zip(self._log10_dmequads, self._log10_dmequad_masks)
                    )
                )
                ** 2
            ) ** 0.5

        @signal_base.cache_call(["delay_params"])
        def get_mean_dm(self, params):
            """Get weighted DMX estimates (distributed to TOAs)."""

            mean_dm = np.zeros(self._ntoas, "d")

            # DMEFAC- and DMJUMP-adjusted
            dm, dme = self.get_dm(params), self.get_dme(params)

            for which in self._dmwhich:
                mean_dm[which] = np.sum(dm[which] / dme[which] ** 2) / np.sum(1.0 / dme[which] ** 2)

            return mean_dm

        @signal_base.cache_call(["delay_params"])
        def get_mean_dme(self, params):
            """Get weighted DMX uncertainties (distributed to TOAs).
            Note that get_phi computes these variances directly."""

            mean_dme = np.zeros(self._ntoas, "d")

            # DMEFAC- and DMJUMP-adjusted
            dme = self.get_dme(params)

            for which in self._dmwhich:
                mean_dme[which] = np.sqrt(1.0 / np.sum(1.0 / dme[which] ** 2))

            return mean_dme

        @signal_base.cache_call(["delay_params"])
        def get_logsignalprior(self, params):
            """Get an additional likelihood/prior term to cover terms that would not
            affect optimization, were they not dependent on DMEFAC, DMEQUAD, and DMJUMP."""

            dm, dme = self.get_dm(params), self.get_dme(params)
            mean_dm, mean_dme = self.get_mean_dm(params), self.get_mean_dme(params)

            # now this is a bit wasteful, because it makes copies of the mean DMX and DMXERR
            # and only uses the first value, but it shouldn't cost us too much
            expterm = -0.5 * np.sum(dm ** 2 / dme ** 2)
            expterm += 0.5 * sum(mean_dm[which][0] ** 2 / mean_dme[which][0] ** 2 for which in self._dmwhich)

            # sum_i [-0.5 * log(dmerr**2)] = -sum_i log dmerr; same for mean_dmerr
            logterm = -np.sum(np.log(dme)) + sum(np.log(mean_dme[which][0]) for which in self._dmwhich)

            return expterm + logterm

        # these are for debugging, but should not enter the likelihood computation

        def get_delta_dm(self, params, use_mean_dm=False):  # DM - DMX
            delta_dm = np.zeros(self._ntoas, "d")

            if use_mean_dm:
                dm = self.get_mean_dm(params)
            else:
                dm = self.get_dm(params)  # DMJUMP-adjusted
            for dmx, which in zip(self._dmx, self._dmwhich):
                delta_dm[which] = dm[which] - (self._dmpar + dmx)

            return delta_dm

        def get_dm_chi2(self, params, use_mean_dm=False):  # 'DM' chi-sqaured
            delta_dm = self.get_delta_dm(params, use_mean_dm=use_mean_dm)

            if use_mean_dm:
                dme = self.get_mean_dme(params)
                chi2 = 0.0
                for idmx, which in enumerate(self._dmwhich):
                    chi2 += (delta_dm[which][0] / dme[which][0]) ** 2

            else:
                dme = self.get_dme(params)  # DMEFAC- and DMEQUAD-adjusted
                chi2 = np.sum((delta_dm / dme) ** 2)

            return chi2

    return WidebandTimingModel


def MarginalizingTimingModel(name="marginalizing_linear_timing_model"):
    basisFunction = utils.normed_tm_basis()

    class TimingModel(signal_base.Signal):
        signal_type = "white noise"
        signal_name = "marginalizing linear timing model"
        signal_id = name

        def __init__(self, psr):
            super(TimingModel, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id

            pname = "_".join([psr.name, name])
            self.Mmat = basisFunction(pname, psr=psr)

            self._params = {}

        @property
        def ndiag_params(self):
            return []

        # there are none, but to be general...
        @signal_base.cache_call("ndiag_params")
        def get_ndiag(self, params):
            return MarginalizingNmat(self.Mmat()[0])

    return TimingModel


class MarginalizingNmat(object):
    def __init__(self, Mmat, Nmat=0):
        self.Mmat, self.Nmat = Mmat, Nmat
        self.Mprior = Mmat.shape[1] * np.log(1e40)

    def __add__(self, other):
        if isinstance(other, MarginalizingNmat):
            raise ValueError("Cannot combine multiple MarginalizingNmat objects.")
        elif isinstance(other, np.ndarray) or hasattr(other, "solve"):
            return MarginalizingNmat(self.Mmat, self.Nmat + other)
        elif other == 0:
            return self
        else:
            raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    # in Python 3.8: @functools.cached_property
    @property
    @functools.lru_cache()
    def cf(self):
        MNM = sps.csc_matrix(self.Nmat.solve(self.Mmat, left_array=self.Mmat))
        return cholesky(MNM)

    @signal_base.simplememobyid
    def MNr(self, res):
        return self.Nmat.solve(res, left_array=self.Mmat)

    @signal_base.simplememobyid
    def MNF(self, T):
        return self.Nmat.solve(T, left_array=self.Mmat)

    @signal_base.simplememobyid
    def MNMMNF(self, T):
        return self.cf(self.MNF(T))

    # we're ignoring logdet = True for two-dimensional cases, but OK
    def solve(self, right, left_array=None, logdet=False):
        if right.ndim == 1 and left_array is right:
            res = right

            rNr, logdet_N = self.Nmat.solve(res, left_array=res, logdet=logdet)

            MNr = self.MNr(res)
            ret = rNr - np.dot(MNr, self.cf(MNr))
            return (ret, logdet_N + self.cf.logdet() + self.Mprior) if logdet else ret
        elif right.ndim == 1 and left_array is not None and left_array.ndim == 2:
            res, T = right, left_array

            TNr = self.Nmat.solve(res, left_array=T)
            return TNr - np.tensordot(self.MNMMNF(T), self.MNr(res), (0, 0))
        elif right.ndim == 2 and left_array is right:
            T = right

            TNT = self.Nmat.solve(T, left_array=T)
            return TNT - np.tensordot(self.MNF(T), self.MNMMNF(T), (0, 0))
        else:
            raise ValueError("Incorrect arguments given to MarginalizingNmat.solve.")
