#utils.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

import scipy.linalg as sl
import scipy.special as ss
import scipy.sparse as sps

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from pkg_resources import resource_filename, Requirement

import enterprise
import enterprise.constants as const
from enterprise.signals.parameter import function

try:
    from sksparse.cholmod import cholesky
except:
    print("You'll need sksparse for get_coefficients() with common signals!")


def get_coefficients(pta,params,n=1,phiinv_method='cliques',
                     common_sparse=False):
    ret = []

    TNrs = pta.get_TNr(params)
    TNTs = pta.get_TNT(params)
    phiinvs = pta.get_phiinv(params, logdet=False,
                             method=phiinv_method)

    # ...repeated code in the two if branches... refactor at will!
    if pta._commonsignals:
        if common_sparse:
            Sigma = sps.block_diag(TNTs,'csc') + sps.csc_matrix(phiinvs)
            TNr = np.concatenate(TNrs)

            ch = cholesky(Sigma)
            mn = ch(TNr)
            Li = sps.linalg.inv(ch.L()).toarray()
        else:
            Sigma = sl.block_diag(*TNTs) + phiinvs
            TNr = np.concatenate(TNrs)

            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, TNr)/s)
            Li = u * np.sqrt(1/s)

        for j in range(n):
            b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

            pardict, ntot = {}, 0
            for i, model in enumerate(pta.pulsarmodels):
                for sig in model._signals:
                    if sig.signal_type in ['basis', 'common basis']:
                        nb = sig.get_basis(params=params).shape[1]

                        if nb + ntot > len(b):
                            raise IndexError("Missing some parameters! "
                                             "You need to disable GP "
                                             "basis column reuse.")

                        pardict[sig.name + '_coefficients'] = b[ntot:nb+ntot]
                        ntot += nb

            if len(ret) <= j:
                ret.append(params.copy())

            ret[j].update(pardict)

        return ret[0] if n is 1 else ret
    else:
        for i, model in enumerate(pta.pulsarmodels):
            phiinv, d, TNT = phiinvs[i], TNrs[i], TNTs[i]

            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            try:
                u, s, _ = sl.svd(Sigma)
                mn = np.dot(u, np.dot(u.T, d)/s)
                Li = u * np.sqrt(1/s)
            except np.linalg.LinAlgError:
                Q, R = sl.qr(Sigma)
                Sigi = sl.solve(R, Q.T)
                mn = np.dot(Sigi, d)
                u, s, _ = sl.svd(Sigi)
                Li = u * np.sqrt(1/s)

            for j in range(n):
                b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

                pardict, ntot = {}, 0
                for sig in model._signals:
                    if sig.signal_type == 'basis':
                        nb = sig.get_basis(params=params).shape[1]

                        if nb + ntot > len(b):
                            raise IndexError("Missing some parameters! "
                                             "You need to disable GP "
                                             "basis column reuse.")

                        pardict[sig.name + '_coefficients'] = b[ntot:nb+ntot]
                        ntot += nb

                if len(ret) <= j:
                    ret.append(params.copy())

                ret[j].update(pardict)

        return ret[0] if n is 1 else ret


class KernelMatrix(np.ndarray):
    def __new__(cls, init):
        if isinstance(init, int):
            ret = np.zeros(init, 'd').view(cls)
        else:
            ret = init.view(cls)

        if ret.ndim == 2:
            ret._cliques = -1 * np.ones(ret.shape[0])
            ret._clcount = 0

        return ret

    # see PTA._setcliques
    def _setcliques(self, idxs):
        allidx = set(self._cliques[idxs])
        maxidx = max(allidx)

        if maxidx == -1:
            self._cliques[idxs] = self._clcount
            self._clcount = self._clcount + 1
        else:
            self._cliques[idxs] = maxidx
            if len(allidx) > 1:
                self._cliques[np.in1d(self._cliques,allidx)] = maxidx

    def add(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] += other
        else:
            if other.ndim == 1:
                self[idx, idx] += other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] += other

        return self

    def set(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] = other
        else:
            if other.ndim == 1:
                self[idx, idx] = other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] = other

        return self

    def inv(self, logdet=False):
        if self.ndim == 1:
            inv = 1.0/self

            if logdet:
                return inv, np.sum(np.log(self))
            else:
                return inv
        else:
            try:
                cf = sl.cho_factor(self)
                inv = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                if logdet:
                    ld = 2.0*np.sum(np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                u, s, v = np.linalg.svd(self)
                inv = np.dot(u/s, u.T)
                if logdet:
                    ld = np.sum(np.log(s))
            if logdet:
                return inv, ld
            else:
                return inv


def create_stabletimingdesignmatrix(designmat, fastDesign=True):
    """
    Stabilize the timing-model design matrix.

    :param designmat: Pulsar timing model design matrix
    :param fastDesign: Stabilize the design matrix the fast way [True]

    :return: Mm: Stabilized timing model design matrix
    """

    Mm = designmat.copy()

    if fastDesign:

        norm = np.sqrt(np.sum(Mm ** 2, axis=0))
        Mm /= norm

    else:

        u, s, v = np.linalg.svd(Mm)
        Mm = u[:, :len(s)]

    return Mm


######################################
# Fourier-basis signal functions #####
######################################


@function
def createfourierdesignmatrix_red(toas, nmodes=30, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  pshift=False, modes=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # add random phase shift to basis functions
    ranphase = (np.random.uniform(0.0, 2 * np.pi, nmodes)
                if pshift else np.zeros(nmodes))

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*toas[:,None]*f[None,:] +
                      ranphase[None,:])
    F[:,1::2] = np.cos(2*np.pi*toas[:,None]*f[None,:] +
                       ranphase[None,:])

    return F, Ffreqs


@function
def createfourierdesignmatrix_dm(toas, freqs, nmodes=30, Tspan=None,
                                 pshift=False, fref=1400, logf=False,
                                 fmin=None, fmax=None, modes=None):
    """
    Construct DM-variation fourier design matrix. Current
    normalization expresses DM signal as a deviation [seconds]
    at fref [MHz]

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param pshift: option to add random phase shift
    :param fref: reference frequency [MHz]
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf,
        fmin=fmin, fmax=fmax, pshift=pshift, modes=modes)

    # compute the DM-variation vectors
    Dm = (fref/freqs)**2

    return F * Dm[:, None], Ffreqs


@function
def createfourierdesignmatrix_env(toas, log10_Amp=-7, log10_Q=np.log10(300),
                                  t0=53000*86400, nmodes=30, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  modes=None):
    """
    Construct fourier design matrix with gaussian envelope.

    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param log10_Amp: log10 of the Amplitude [s]
    :param t0: mean of gaussian envelope [s]
    :param log10_Q: log10 of standard deviation of gaussian envelope [days]
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix with gaussian envelope
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf,
        fmin=fmin, fmax=fmax, modes=modes)

    # compute gaussian envelope
    A = 10**log10_Amp
    Q = 10**log10_Q * 86400
    env = A * np.exp(-(toas-t0)**2/2/Q**2)
    return F * env[:, None], Ffreqs


@function
def createfourierdesignmatrix_ephem(toas, pos, nmodes=30, Tspan=None):
    """
    Construct ephemeris perturbation Fourier design matrix and frequencies.
    The matrix contains nmodes*6 columns, ordered as by frequency first,
    Cartesian coordinate second:

    sin(f0) [x], sin(f0) [y], sin(f0) [z],
    cos(f0) [x], cos(f0) [y], cos(f0) [z],
    sin(f1) [x], sin(f1) [y], sin(f1) [z], ...

    The corresponding frequency vector repeats every entry six times.
    This design matrix should be used with monopole_orf and with
    a powerlaw that specifies components=6.

    :param toas: vector of time series in seconds
    :param pos: pulsar position as Cartesian vector
    :param nmodes: number of Fourier coefficients
    :param Tspan: Tspan used to define Fourier bins

    :return: F: Fourier design matrix of shape (len(toas),6*nmodes)
    :return: f: Sampling frequencies (6*nmodes)
    """

    F0, F0f = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan)

    F1 = np.zeros((len(toas),nmodes,2,3), 'd')
    F1[:,:,0,:] = F0[:,0::2,np.newaxis]
    F1[:,:,1,:] = F0[:,1::2,np.newaxis]

    # verify this is the scalar product we want
    F1 *= pos

    F1f = np.zeros((nmodes,2,3), 'd')
    F1f[:,:,:] = F0f[::2,np.newaxis,np.newaxis]

    return F1.reshape((len(toas),nmodes*6)), F1f.reshape((nmodes*6,))


def createfourierdesignmatrix_eph(t, nmodes, phi, theta, freq=False,
                                  Tspan=None, logf=False, fmin=None,
                                  fmax=None, modes=None):
    raise NotImplementedError(
        "createfourierdesignmatrix_eph was removed, " +
        "and replaced with createfourierdesignmatrix_ephem")


###################################
# Deterministic GW signal functions
###################################


def make_ecc_interpolant():

    """
    Make interpolation function from eccentricity file to
    determine number of harmonics to use for a given
    eccentricity.

    :returns: interpolant
    """

    pth = resource_filename(Requirement.parse('libstempo'),
                            'libstempo/ecc_vs_nharm.txt')

    fil = np.loadtxt(pth)

    return interp1d(fil[:,0], fil[:,1])


# get interpolant for eccentric binaries
ecc_interp = make_ecc_interpolant()


def get_edot(F, mc, e):

    """
    Compute eccentricity derivative from Taylor et al. (2016)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: de/dt
    """

    # chirp mass
    mc *= const.Tsun

    dedt = -304/(15*mc) * (2*np.pi*mc*F)**(8/3) * e * \
        (1 + 121/304*e**2) / ((1-e**2)**(5/2))

    return dedt


def get_Fdot(F, mc, e):
    """
    Compute frequency derivative from Taylor et al. (2016)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: dF/dt
    """

    # chirp mass
    mc *= const.Tsun

    dFdt = 48 / (5*np.pi*mc**2) * (2*np.pi*mc*F)**(11/3) * \
        (1 + 73/24*e**2 + 37/96*e**4) / ((1-e**2)**(7/2))

    return dFdt


def get_gammadot(F, mc, q, e):
    """
    Compute gamma dot from Barack and Cutler (2004)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param e: Eccentricity of binary

    :returns: dgamma/dt
    """

    # chirp mass
    mc *= const.Tsun

    # total mass
    m = (((1+q)**2)/q)**(3/5) * mc

    dgdt = 6*np.pi*F * (2*np.pi*F*m)**(2/3) / (1-e**2) * \
        (1 + 0.25*(2*np.pi*F*m)**(2/3)/(1-e**2)*(26-15*e**2))

    return dgdt


def get_coupled_constecc_eqns(y, t, mc, e0):
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:

    F: Orbital frequency [Hz]
    phase0: Orbital phase [rad]

    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]

    :returns: array of derivatives [dF/dt, dphase/dt]
    """

    F = y[0]

    dFdt = get_Fdot(F, mc, e0)
    dphasedt = 2*np.pi*F

    return np.array([dFdt, dphasedt])


def get_coupled_ecc_eqns(y, t, mc, q):
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:

    F: Orbital frequency [Hz]
    e: Orbital eccentricity
    gamma: Angle of precession of periastron [rad]
    phase0: Orbital phase [rad]

    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary

    :returns: array of derivatives [dF/dt, de/dt, dgamma/dt, dphase/dt]
    """

    F = y[0]
    e = y[1]

    dFdt = get_Fdot(F, mc, e)
    dedt = get_edot(F, mc, e)
    dgdt = get_gammadot(F, mc, q, e)
    dphasedt = 2*np.pi*F

    return np.array([dFdt, dedt, dgdt, dphasedt])


def solve_coupled_constecc_solution(F0, e0, phase0, mc, t):
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at
    a given time.

    :param F0: Initial orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param t: Time at which to evaluate solution [s]

    :returns: (F(t), phase(t))
    """

    y0 = np.array([F0, phase0])

    y, infodict = odeint(get_coupled_constecc_eqns, y0, t,
                         args=(mc,e0), full_output=True)

    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0

    return ret


def solve_coupled_ecc_solution(F0, e0, gamma0, phase0, mc, q, t):
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at
    a given time.

    :param F0: Initial orbital frequency [Hz]
    :param e0: Initial orbital eccentricity
    :param gamma0: Initial angle of precession of periastron [rad]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param t: Time at which to evaluate solution [s]

    :returns: (F(t), e(t), gamma(t), phase(t))
    """

    y0 = np.array([F0, e0, gamma0, phase0])

    y, infodict = odeint(get_coupled_ecc_eqns, y0, t,
                         args=(mc,q), full_output=True)

    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0

    return ret


def get_an(n, mc, dl, h0, F, e):
    """
    Compute a_n from Eq. 22 of Taylor et al. (2016).

    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity

    :returns: a_n
    """

    # convert to seconds
    mc *= const.Tsun
    dl *= const.Mpc / const.c

    omega = 2 * np.pi * F

    if h0 is None:
        amp = n * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = n * h0 / 2.0

    ret = -amp * (ss.jn(n-2,n*e) - 2*e*ss.jn(n-1,n*e) +
                  (2/n)*ss.jn(n,n*e) + 2*e*ss.jn(n+1,n*e) -
                  ss.jn(n+2,n*e))

    return ret


def get_bn(n, mc, dl, h0, F, e):
    """
    Compute b_n from Eq. 22 of Taylor et al. (2015).

    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity

    :returns: b_n
    """

    # convert to seconds
    mc *= const.Tsun
    dl *= const.Mpc / const.c

    omega = 2 * np.pi * F

    if h0 is None:
        amp = n * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = n * h0 / 2.0

    ret = (-amp * np.sqrt(1-e**2) * (ss.jn(n-2,n*e) -
           2*ss.jn(n,n*e) + ss.jn(n+2,n*e)))

    return ret


def get_cn(n, mc, dl, h0, F, e):
    """
    Compute c_n from Eq. 22 of Taylor et al. (2016).

    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity

    :returns: c_n
    """

    # convert to seconds
    mc *= const.Tsun
    dl *= const.Mpc / const.c

    omega = 2 * np.pi * F

    if h0 is None:
        amp = 2 * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = h0

    ret = amp * ss.jn(n,n*e) / (n * omega)

    return ret


def calculate_splus_scross(nmax, mc, dl, h0, F, e,
                           t, l0, gamma, gammadot, inc):
    """
    Calculate splus and scross for a CGW summed over all harmonics.
    This waveform differs slightly from that in Taylor et al (2016)
    in that it includes the time dependence of the advance of periastron.

    :param nmax: Total number of harmonics to use
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: TOAs [s]
    :param l0: Initial eccentric anomoly [rad]
    :param gamma: Angle of periastron advance [rad]
    :param gammadot: Time derivative of angle of periastron advance [rad/s]
    :param inc: Inclination angle [rad]

    :return splus, scross: plus and cross time-domain waveforms for a CGW
    """
    n = np.arange(1, nmax)

    # time dependent amplitudes
    an = get_an(n, mc, dl, h0, F, e)
    bn = get_bn(n, mc, dl, h0, F, e)
    cn = get_cn(n, mc, dl, h0, F, e)

    # time dependent terms
    omega = 2*np.pi*F
    gt = gamma + gammadot * t
    lt = l0 + omega * t

    # tiled phase
    phase1 = n * np.tile(lt, (nmax-1,1)).T
    phase2 = np.tile(gt, (nmax-1,1)).T

    sinp1 = np.sin(phase1)
    cosp1 = np.cos(phase1)
    sinp2 = np.sin(2*phase2)
    cosp2 = np.cos(2*phase2)

    sinpp = sinp1*cosp2 + cosp1*sinp2
    cospp = cosp1*cosp2 - sinp1*sinp2
    sinpm = sinp1*cosp2 - cosp1*sinp2
    cospm = cosp1*cosp2 + sinp1*sinp2

    # intermediate terms
    sp = (sinpm/(n*omega-2*gammadot) +
          sinpp/(n*omega+2*gammadot))
    sm = (sinpm/(n*omega-2*gammadot) -
          sinpp/(n*omega+2*gammadot))
    cp = (cospm/(n*omega-2*gammadot) +
          cospp/(n*omega+2*gammadot))
    cm = (cospm/(n*omega-2*gammadot) -
          cospp/(n*omega+2*gammadot))

    splus_n = (-0.5 * (1+np.cos(inc)**2) * (an*sp - bn*sm) +
               (1-np.cos(inc)**2)*cn * sinp1)
    scross_n = np.cos(inc) * (an*cm - bn*cp)

    return np.sum(splus_n, axis=1), np.sum(scross_n, axis=1)


def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param pos: Unit vector from Earth to pulsar
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians

    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi),
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi),
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])

    fplus = (0.5 * (np.dot(m, pos)**2 - np.dot(n, pos)**2) /
             (1+np.dot(omhat, pos)))
    fcross = (np.dot(m, pos)*np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu


@function
def bwm_delay(toas, pos, log10_h=-14.0, cos_gwtheta=0.0, gwphi=0.0,
              gwpol=0.0, t0=55000, antenna_pattern_fn=None):
    """
    Function that calculates the earth-term gravitational-wave
    burst-with-memory signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.
    This version uses the F+/Fx polarization modes, as verified with the
    Continuous Wave and Anisotropy papers.

    :param toas: Time-of-arrival measurements [s]
    :param pos: Unit vector from Earth to pulsar
    :param log10_h: log10 of GW strain
    :param cos_gwtheta: Cosine of GW polar angle
    :param gwphi: GW azimuthal polar angle [rad]
    :param gwpol: GW polarization angle
    :param t0: Burst central time [day]
    :param antenna_pattern_fn:
        User defined function that takes `pos`, `gwtheta`, `gwphi` as
        arguments and returns (fplus, fcross)

    :return: the waveform as induced timing residuals (seconds)
    """

    # convert
    h = 10**log10_h
    gwtheta = np.arccos(cos_gwtheta)
    t0 *= const.day

    # antenna patterns
    if antenna_pattern_fn is None:
        apc = create_gw_antenna_pattern(pos, gwtheta, gwphi)
    else:
        apc = antenna_pattern_fn(pos, gwtheta, gwphi)

    # grab fplus, fcross
    fp, fc = apc[0], apc[1]

    # combined polarization
    pol = np.cos(2*gwpol)*fp + np.sin(2*gwpol)*fc

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time-series for the pulsar
    return pol * h * heaviside(toas-t0) * (toas-t0)


@function
def create_quantization_matrix(toas, dt=1, nmin=2):
    """Create quantization matrix mapping TOAs to observing epochs."""
    isort = np.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    # find only epochs with more than 1 TOA
    bucket_ind2 = [ind for ind in bucket_ind if len(ind) >= nmin]

    U = np.zeros((len(toas),len(bucket_ind2)),'d')
    for i,l in enumerate(bucket_ind2):
        U[l,i] = 1

    weights = np.ones(U.shape[1])

    return U, weights


def quant2ind(U):
    """
    Use quantization matrix to return slices of non-zero elements.

    :param U: quantization matrix

    :return: list of `slice`s for non-zero elements of U

    .. note:: This function assumes that the pulsar TOAs were sorted by time.

    """
    inds = []
    for cc, col in enumerate(U.T):
        epinds = np.flatnonzero(col)
        if epinds[-1] - epinds[0] + 1 != len(epinds):
            raise ValueError('ERROR: TOAs not sorted properly!')
        inds.append(slice(epinds[0], epinds[-1]+1))
    return inds


def linear_interp_basis(toas, dt=30*86400):
    """Provides a basis for linear interpolation.

    :param toas: Pulsar TOAs in seconds
    :param dt: Linear interpolation step size in seconds.

    :returns: Linear interpolation basis and nodes
    """

    # evenly spaced points
    x = np.arange(toas.min(), toas.max()+dt, dt)
    M = np.zeros((len(toas), len(x)))

    # make linear interpolation basis
    for ii in range(len(x)-1):
        idx = np.logical_and(toas >= x[ii], toas <= x[ii+1])
        M[idx, ii] = (toas[idx] - x[ii+1]) / (x[ii] - x[ii+1])
        M[idx, ii+1] = (toas[idx] - x[ii]) / (x[ii+1] - x[ii])

    # only return non-zero columns
    idx = M.sum(axis=0) != 0

    return M[:, idx], x[idx]


@function
def powerlaw(f, log10_A=-16, gamma=5, components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    return ((10**log10_A)**2 / 12.0 / np.pi**2 *
            const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, components))


@function
def turnover(f, log10_A=-15, gamma=4.33, lf0=-8.5, kappa=10/3, beta=0.5):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    hcf = (10**log10_A * (f / const.fyr) ** ((3-gamma) / 2) /
           (1 + (10**lf0 / f) ** kappa) ** beta)
    return hcf**2/12/np.pi**2/f**3*np.repeat(df, 2)


# overlap reduction functions

@function
def hd_orf(pos1, pos2):
    """Hellings & Downs spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5


@function
def dipole_orf(pos1, pos2):
    """Dipole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1 + 1e-5
    else:
        return np.dot(pos1, pos2)


@function
def monopole_orf(pos1, pos2):
    """Monopole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1.0 + 1e-5
    else:
        return 1.0


@function
def anis_orf(pos1, pos2, params, **kwargs):
    """Anisotropic GWB spatial correlation function."""

    anis_basis = kwargs['anis_basis']
    psrs_pos = kwargs['psrs_pos']
    lmax = kwargs['lmax']

    psr1_index = [ii for ii in range(len(psrs_pos))
                  if np.all(psrs_pos[ii] == pos1)][0]
    psr2_index = [ii for ii in range(len(psrs_pos))
                  if np.all(psrs_pos[ii] == pos2)][0]

    clm = np.zeros((lmax+1)**2)
    clm[0] = 2.0*np.sqrt(np.pi)
    if lmax > 0:
        clm[1:] = params

    return sum(clm[ii]*basis for ii,basis
               in enumerate(anis_basis[:(lmax+1)**2,
                                       psr1_index, psr2_index]))


@function
def unnormed_tm_basis(Mmat):
    return Mmat, np.ones_like(Mmat.shape[1])


@function
def normed_tm_basis(Mmat, norm=None):
    if norm is None:
        norm = np.sqrt(np.sum(Mmat**2, axis=0))

    nmat = Mmat / norm
    nmat[:,norm == 0] = 0

    return nmat, np.ones_like(Mmat.shape[1])


@function
def svd_tm_basis(Mmat):
    u, s, v = np.linalg.svd(Mmat, full_matrices=False)
    return u, np.ones_like(s)


@function
def tm_prior(weights):
    return weights * 1e40


# Physical ephemeris model utility functions

t_offset = 55197.0
e_ecl = 23.43704 * np.pi / 180.0
M_ecl = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(e_ecl), -np.sin(e_ecl)],
                  [0.0, np.sin(e_ecl), np.cos(e_ecl)]])


def get_planet_orbital_elements():
    """Grab physical ephemeris model files"""
    dpath = enterprise.__path__[0] + '/datafiles/ephemeris/'
    jup_mjd = np.load(dpath + 'jupiter-orbel-mjd.npy')
    jup_orbelxyz = np.load(dpath + 'jupiter-orbel-xyz-svd.npy')
    sat_mjd = np.load(dpath + 'saturn-orbel-mjd.npy')
    sat_orbelxyz = np.load(dpath + 'saturn-orbel-xyz-svd.npy')
    return jup_mjd, jup_orbelxyz, sat_mjd, sat_orbelxyz


def ecl2eq_vec(x):
    """
    Rotate (n,3) vector time series from ecliptic to equatorial.
    """
    return np.einsum('jk,ik->ij', M_ecl, x)


def eq2ecl_vec(x):
    """
    Rotate (n,3) vector time series from equatorial to ecliptic.
    """
    return np.einsum('kj,ik->ij', M_ecl, x)


def euler_vec(z, y, x, n):
    """
    Return (n,3,3) tensor with each (3,3) block containing an
    Euler rotation with angles z, y, x. Optionally each of z, y, x
    can be a vector of length n.
    """
    L = np.zeros((n,3,3), 'd')
    cosx, sinx = np.cos(x), np.sin(x)
    L[:,0,0] = 1
    L[:,1,1] = L[:,2,2] = cosx
    L[:,1,2] = -sinx
    L[:,2,1] = sinx

    N = np.zeros((n,3,3),'d')
    cosy, siny = np.cos(y), np.sin(y)
    N[:,0,0] = N[:,2,2] = cosy
    N[:,1,1] = 1
    N[:,0,2] = siny
    N[:,2,0] = -siny

    ret = np.einsum('ijk,ikl->ijl', L, N)

    M = np.zeros((n,3,3),'d')
    cosz, sinz = np.cos(z), np.sin(z)
    M[:,0,0] = M[:,1,1] = cosz
    M[:,0,1] = -sinz
    M[:,1,0] = sinz
    M[:,2,2] = 1

    ret = np.einsum('ijk,ikl->ijl', ret, M)

    return ret


def ss_framerotate(mjd, planet, x, y, z, dz,
                   offset=None, equatorial=False):
    """
    Rotate planet trajectory given as (n,3) tensor,
    by ecliptic Euler angles x, y, z, and by z rate
    dz. The rate has units of rad/year, and is referred
    to offset 2010/1/1. dates must be given in MJD.
    """
    if equatorial:
        planet = eq2ecl_vec(planet)

    E = euler_vec(z + dz * (mjd - t_offset) / 365.25, y, x,
                  planet.shape[0])

    planet = np.einsum('ijk,ik->ij', E, planet)

    if offset is not None:
        planet = np.array(offset) + planet

    if equatorial:
        planet = ecl2eq_vec(planet)

    return planet


def dmass(planet, dm_over_Msun):
    return dm_over_Msun * planet


@function
def physicalephem_spectrum(sigmas):
    # note the creative use of the "labels" (the very sigmas, not frequencies)
    return sigmas**2


@function
def createfourierdesignmatrix_physicalephem(toas, planetssb, pos_t,
                                            frame_drift_rate=1e-9,
                                            d_jupiter_mass=1.54976690e-11,
                                            d_saturn_mass=8.17306184e-12,
                                            d_uranus_mass=5.71923361e-11,
                                            d_neptune_mass=7.96103855e-11,
                                            jup_orb_elements=0.05,
                                            sat_orb_elements=0.5):
    """
    Construct physical ephemeris perturbation design matrix and 'frequencies'.
    Parameters can be excluded by setting the corresponding prior sigma to None

    :param toas:             vector of time series in seconds
    :param pos:              pulsar position as Cartesian vector
    :param frame_drift_rate: normal sigma for frame drift rate
    :param d_jupiter_mass:   normal sigma for Jupiter mass perturbation
    :param d_saturn_mass:    normal sigma for Saturn mass perturbation
    :param d_uranus_mass:    normal sigma for Uranus mass perturbation
    :param d_neptune_mass:   normal sigma for Neptune mass perturbation
    :param jup_orb_elements: normal sigma for Jupiter orbital elem. perturb.
    :param sat_orb_elements: normal sigma for Saturn orbital elem. perturb.

    :return: F: Fourier design matrix of shape (len(toas), nvecs)
    :return: sigmas: Phi sigmas (nvecs, to be passed to physicalephem_spectrum)
    """

    # Jupiter + Saturn orbit definitions that we pass to physical_ephem_delay
    oa = {'inc_jupiter_orb': True, 'inc_saturn_orb': True}
    oa['jup_mjd'], oa['jup_orbelxyz'], oa['sat_mjd'], oa['sat_orbelxyz'] = \
        get_planet_orbital_elements()

    dpar = 1e-3  # may need finessing
    Fl, Phil = [], []

    for parname in ['frame_drift_rate',
                    'd_jupiter_mass', 'd_saturn_mass',
                    'd_uranus_mass', 'd_neptune_mass',
                    'jup_orb_elements', 'sat_orb_elements']:

        ppar = locals()[parname]
        if ppar:
            if parname not in ['jup_orb_elements', 'sat_orb_elements']:
                # need to normalize?
                Fl.append(physical_ephem_delay(toas, planetssb, pos_t,
                                               **{parname: dpar})/dpar)
                Phil.append(ppar)
            else:
                for i in range(6):
                    c = np.zeros(6)
                    c[i] = dpar

                    Fl.append(physical_ephem_delay(toas, planetssb, pos_t,
                                                   **{parname: c}, **oa)/dpar)
                    Phil.append(ppar)

    return np.array(Fl).T.copy(), np.array(Phil)


@function
def physical_ephem_delay(toas, planetssb, pos_t, frame_drift_rate=0,
                         d_jupiter_mass=0, d_saturn_mass=0, d_uranus_mass=0,
                         d_neptune_mass=0, jup_orb_elements=np.zeros(6),
                         sat_orb_elements=np.zeros(6), inc_jupiter_orb=False,
                         jup_orbelxyz=None, jup_mjd=None, inc_saturn_orb=False,
                         sat_orbelxyz=None, sat_mjd=None, equatorial=True):

        # convert toas to MJD
        mjd = toas / 86400

        # grab planet-to-SSB vectors
        earth = planetssb[:, 2, :3]
        jupiter = planetssb[:, 4, :3]
        saturn = planetssb[:, 5, :3]
        uranus = planetssb[:, 6, :3]
        neptune = planetssb[:, 7, :3]

        # do frame rotation
        earth = ss_framerotate(mjd, earth, 0.0, 0.0, 0.0, frame_drift_rate,
                               offset=None, equatorial=equatorial)

        # mass perturbations
        mpert = [(jupiter, d_jupiter_mass), (saturn, d_saturn_mass),
                 (uranus, d_uranus_mass), (neptune, d_neptune_mass)]
        for planet, dm in mpert:
            earth += dmass(planet, dm)

        # jupter orbital element perturbations
        if inc_jupiter_orb:
            jup_perturb_tmp = 0.0009547918983127075 * np.einsum(
                'i,ijk->jk', jup_orb_elements, jup_orbelxyz)
            earth += np.array([np.interp(mjd, jup_mjd, jup_perturb_tmp[:,aa])
                               for aa in range(3)]).T

        # saturn orbital element perturbations
        if inc_saturn_orb:
            sat_perturb_tmp = 0.00028588567008942334 * np.einsum(
                'i,ijk->jk', sat_orb_elements, sat_orbelxyz)
            earth += np.array([np.interp(mjd, sat_mjd, sat_perturb_tmp[:,aa])
                               for aa in range(3)]).T

        # construct the true geocenter to barycenter roemer
        tmp_roemer = np.einsum('ij,ij->i', planetssb[:, 2, :3], pos_t)

        # create the delay
        delay = tmp_roemer - np.einsum('ij,ij->i', earth, pos_t)

        return delay
