# anis_coefficients.py

import healpy as hp
import numpy as np
import scipy.special as ss


"""
Script to compute the correlation basis-functions for various anisotropic
configurations of the GW background energy-density

-- Rutger van Haasteren (June 2014)
-- Stephen Taylor (modifications, February 2016)

"""


def real_sph_harm(mm, ll, phi, theta):
    """
    The real-valued spherical harmonics.
    """
    if mm > 0:
        ans = (1.0 / np.sqrt(2)) * (ss.sph_harm(mm, ll, phi, theta) + ((-1) ** mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm == 0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm < 0:
        ans = (1.0 / (np.sqrt(2) * complex(0.0, 1))) * (
            ss.sph_harm(-mm, ll, phi, theta) - ((-1) ** mm) * ss.sph_harm(mm, ll, phi, theta)
        )

    return ans.real


def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a):
    """
    Create the signal response matrix FAST
    """

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta)


def createSignalResponse(pphi, ptheta, gwphi, gwtheta):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW propagation direction
    @param gwtheta: Theta of GW propagation direction

    @return:    Signal response matrix of Earth-term
    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=False)

    # Pixel maps are lumped together, polarization pixels are neighbours
    F = np.zeros((Fp.shape[0], 2 * Fp.shape[1]))
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F


def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW propagation direction
    @param gwtheta: Theta of GW propagation direction
    @param plus:    Whether or not this is the plus-polarization
    @param norm:    Normalise the correlations to equal Jenet et. al (2005)

    @return:    Signal response matrix of Earth-term
    """
    # Create the unit-direction vectors. First dimension
    # will be collapsed later. Sign convention of Gair et al. (2014)
    Omega = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])

    mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
    nhat = np.array([-np.cos(gwphi) * np.cos(gwtheta), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)])

    p = np.array([np.cos(pphi) * np.sin(ptheta), np.sin(pphi) * np.sin(ptheta), np.cos(ptheta)])

    # There is a factor of 3/2 difference between the Hellings & Downs
    # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # et al. 2014). This factor 'normalises' the correlation matrix.
    npixels = Omega.shape[2]
    if norm:
        # Add extra factor of 3/2
        c = np.sqrt(1.5) / np.sqrt(npixels)
    else:
        c = 1.0 / np.sqrt(npixels)

    # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
    # al. (2014), with right-handed coordinate system
    if plus:
        # The sum over axis=0 represents an inner-product
        Fsig = (
            0.5 * c * (np.sum(nhat * p, axis=0) ** 2 - np.sum(mhat * p, axis=0) ** 2) / (1 - np.sum(Omega * p, axis=0))
        )
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / (1 - np.sum(Omega * p, axis=0))

    return Fsig


def almFromClm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
    """
    maxl = int(np.sqrt(len(clm))) - 1

    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl + 1):
        for mm in range(-ll, ll + 1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))

            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)

            clmindex += 1

    return alm


def clmFromAlm(alm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
    """
    nalm = len(alm)
    maxl = int(np.sqrt(9.0 - 4.0 * (2.0 - 2.0 * nalm)) * 0.5 - 1.5)  # Really?
    nclm = (maxl + 1) ** 2

    # Check the solution. Went wrong one time..
    if nalm != int(0.5 * (maxl + 1) * (maxl + 2)):
        raise ValueError("Check numerical precision. This should not happen")

    clm = np.zeros(nclm)

    clmindex = 0
    for ll in range(0, maxl + 1):
        for mm in range(-ll, ll + 1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))

            if mm == 0:
                clm[clmindex] = alm[almindex].real
            elif mm < 0:
                clm[clmindex] = -alm[almindex].imag * np.sqrt(2)
            elif mm > 0:
                clm[clmindex] = alm[almindex].real * np.sqrt(2)

            clmindex += 1

    return clm


def mapFromClm_fast(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm))) - 1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h


def mapFromClm(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use real_sph_harm for the map
    """
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)

    h = np.zeros(npixels)

    ind = 0
    maxl = int(np.sqrt(len(clm))) - 1
    for ll in range(maxl + 1):
        for mm in range(-ll, ll + 1):
            h += clm[ind] * real_sph_harm(mm, ll, pixels[1], pixels[0])
            ind += 1

    return h


def clmFromMap_fast(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values

    Use Healpix spherical harmonics for computational efficiency
    """
    alm = hp.sphtfunc.map2alm(h, lmax=lmax)
    alm[0] = np.sum(h) * np.sqrt(4 * np.pi) / len(h)

    return clmFromAlm(alm)


def clmFromMap(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values

    Use real_sph_harm for the map
    """
    npixels = len(h)
    nside = hp.npix2nside(npixels)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)

    clm = np.zeros((lmax + 1) ** 2)

    ind = 0
    for ll in range(lmax + 1):
        for mm in range(-ll, ll + 1):
            clm[ind] += np.sum(h * real_sph_harm(mm, ll, pixels[1], pixels[0]))
            ind += 1

    return clm * 4 * np.pi / npixels


def getCov(clm, nside, F_e):
    """
    Given a vector of clm values, construct the covariance matrix

    @param clm:     Array with Clm values
    @param nside:   Healpix nside resolution
    @param F_e:     Signal response matrix

    @return:    Cross-pulsar correlation for this array of clm values
    """
    # Create a sky-map (power)
    # Use mapFromClm to compare to real_sph_harm. Fast uses Healpix
    # sh00 = mapFromClm(clm, nside)
    sh00 = mapFromClm_fast(clm, nside)

    # Double the power (one for each polarization)
    sh = np.array([sh00, sh00]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))


def anis_basis(psr_locs, lmax, nside=32):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations

    @param psr_locs:    Location of the pulsars [phi, theta]
    @param lmax:        Maximum l to go up to
    @param nside:       What nside to use in the pixelation [32]

    Note: GW directions are in direction of GW propagation
    """
    pphi = psr_locs[:, 0]
    ptheta = psr_locs[:, 1]

    # Create the pixels
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]

    # Create the signal response matrix
    F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)

    # Loop over all (l,m)
    basis = []
    nclm = (lmax + 1) ** 2
    clmindex = 0
    for ll in range(0, lmax + 1):
        for mm in range(-ll, ll + 1):
            clm = np.zeros(nclm)
            clm[clmindex] = 1.0

            basis.append(getCov(clm, nside, F_e))
            clmindex += 1

    return np.array(basis)


def orfFromMap_fast(psr_locs, usermap, response=None):
    """
    Calculate an ORF from a user-defined sky map.

    @param psr_locs:    Location of the pulsars [phi, theta]
    @param usermap:     Provide a healpix map for GW power

    Note: GW directions are in direction of GW propagation
    """
    if response is None:
        pphi = psr_locs[:, 0]
        ptheta = psr_locs[:, 1]

        # Create the pixels
        nside = hp.npix2nside(len(usermap))
        npixels = hp.nside2npix(nside)
        pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
        gwtheta = pixels[0]
        gwphi = pixels[1]

        # Create the signal response matrix
        F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)
    elif response is not None:
        F_e = response

    # Double the power (one for each polarization)
    sh = np.array([usermap, usermap]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))
