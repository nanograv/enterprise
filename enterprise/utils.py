#utils.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import special as ss
from pkg_resources import resource_filename, Requirement
import enterprise.constants as const


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
        Mm = u[:,:len(s)]

    return Mm


######################################
# Fourier-basis signal functions #####
######################################


def createfourierdesignmatrix_red(t, nmodes, freq=False, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  pshift=False):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift

    :return: F: fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    :return: ranphase: Phase offsets applied to basis functions
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    # add random phase shift to basis functions
    if pshift:
        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    elif not pshift:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # The sine/cosine modes
    if pshift:
        F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:] +
                          ranphase[None,:])
        F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:] +
                           ranphase[None,:])
    elif not pshift:
        F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:])
        F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:])

    if freq:
        return F, Ffreqs, ranphase
    else:
        return F, ranphase


def createfourierdesignmatrix_dm(t, nmodes, ssbfreqs, freq=False,
                                 Tspan=None, logf=False, fmin=None,
                                 fmax=None):

    """
    Construct DM-variation fourier design matrix.

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param ssbfreqs: radio frequencies of observations [MHz]
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # compute the DM-variation vectors
    Dm = 1.0/(const.DM_K * ssbfreqs**2.0 * 1e12)

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:]) * Dm[:,None]
    F[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:]) * Dm[:,None]

    if freq:
        return F, Ffreqs
    else:
        return F


def createfourierdesignmatrix_eph(t, nmodes, phi, theta, freq=False,
                                  Tspan=None, logf=False, fmin=None,
                                  fmax=None):

    """
    Construct ephemeris fourier design matrix.

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param phi: azimuthal coordinate of pulsar
    :param theta: polar coordinate of pulsar
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: Fx: x-axis ephemeris fourier design matrix
    :return: Fy: y-axis ephemeris fourier design matrix
    :return: Fz: z-axis ephemeris fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    """

    N = len(t)
    Fx = np.zeros((N, 2*nmodes))
    Fy = np.zeros((N, 2*nmodes))
    Fz = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # define the pulsar position vector
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    # The sine/cosine modes
    Fx[:,::2] = np.sin(2*np.pi*t[:,None]*f[None,:])
    Fx[:,1::2] = np.cos(2*np.pi*t[:,None]*f[None,:])

    Fy = Fx.copy()
    Fz = Fx.copy()

    Fx *= x
    Fy *= y
    Fz *= z

    if freq:
        return Fx, Fy, Fz, Ffreqs
    else:
        return Fx, Fy, Fz


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


def fplus_fcross(psr, gwtheta, gwphi):
    """
    Compute gravitational-wave quadrupolar antenna pattern.

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]

    :returns: fplus, fcross
    """

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    # pulsar location
    ptheta = np.pi/2 - psr.psr_locs[1]
    pphi = psr.psr_locs[0]

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),
                     np.cos(ptheta)])

    fplus = (0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) /
             (1+np.dot(omhat, phat)))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))

    return fplus, fcross
