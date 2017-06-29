#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_likelihood
----------------------------------

Tests of likelihood module
"""


import unittest
import numpy as np
import scipy.linalg as sl

from tests.enterprise_test_data import datadir
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils


def get_noise_from_pal2(noisefile):
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        elif 'equad' in line:
            par = 'log10_equad'
            flag = ln[0].split('equad-')[-1]
        elif 'jitter_q' in line:
            par = 'log10_ecorr'
            flag = ln[0].split('jitter_q-')[-1]
        elif 'RN-Amplitude' in line:
            par = 'log10_A'
            flag = ''
        elif 'RN-spectral-index' in line:
            par = 'gamma'
            flag = ''
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params


class TestLikelihood(unittest.TestCase):

    def setUp(self):
        """Setup the Pulsar object."""

        # initialize Pulsar class
        self.psrs = [Pulsar(datadir + '/B1855+09_NANOGrav_9yv1.gls.par',
                            datadir + '/B1855+09_NANOGrav_9yv1.tim'),
                     Pulsar(datadir + '/J1909-3744_NANOGrav_9yv1.gls.par',
                            datadir + '/J1909-3744_NANOGrav_9yv1.tim')]

    def compute_like(self, npsrs=1, inc_corr=False):

        # get parameters from PAL2 style noise files
        params = get_noise_from_pal2(datadir+'/B1855+09_noise.txt')
        params2 = get_noise_from_pal2(datadir+'/J1909-3744_noise.txt')
        params.update(params2)

        psrs = self.psrs if npsrs == 2 else [self.psrs[0]]

        if inc_corr:
            params.update({'GW_gamma': 4.33, 'GW_log10_A':-15.0})

        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in psrs]
        tmax = [p.toas.max() for p in psrs]
        Tspan = np.max(tmax) - np.min(tmin)

        # setup basic model
        efac = parameter.Constant()
        equad = parameter.Constant()
        ecorr = parameter.Constant()
        log10_A = parameter.Constant()
        gamma = parameter.Constant()

        selection = Selection(selections.by_backend)

        ef = white_signals.MeasurementNoise(efac=efac,
                                            selection=selection)
        eq = white_signals.EquadNoise(log10_equad=equad,
                                      selection=selection)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,
                                            selection=selection)

        pl = signal_base.Function(utils.powerlaw, log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl)

        orf = signal_base.Function(utils.hd_orf)
        crn = gp_signals.FourierBasisCommonGP(pl, orf, components=20,
                                              name='GW', Tspan=Tspan)

        tm = gp_signals.TimingModel()

        if inc_corr:
            s = ef + eq + ec + rn + crn + tm
        else:
            s = ef + eq + ec + rn + tm

        pta = signal_base.PTA([s(psr) for psr in psrs])

        # set parameters
        pta.set_default_params(params)

        # get parameters
        efacs, equads, ecorrs, log10_A, gamma = [], [], [], [], []
        for pname in [p.name for p in psrs]:
            efacs.append([params[key] for key in sorted(params.keys())
                          if 'efac' in key and pname in key])
            equads.append([params[key] for key in sorted(params.keys())
                           if 'equad' in key and pname in key])
            ecorrs.append([params[key] for key in sorted(params.keys())
                           if 'ecorr' in key and pname in key])
            log10_A.append(params['{}_log10_A'.format(pname)])
            gamma.append(params['{}_gamma'.format(pname)])
        GW_gamma = 4.33
        GW_log10_A = -15.0

        # correct value
        tflags = [sorted(list(np.unique(p.backend_flags))) for p in psrs]
        cfs, logdets, phis, Ts = [], [], [], []
        for ii, (psr, flags) in enumerate(zip(psrs, tflags)):
            nvec0 = np.zeros_like(psr.toas)
            for ct, flag in enumerate(flags):
                ind = psr.backend_flags == flag
                nvec0[ind] = efacs[ii][ct]**2 * psr.toaerrs[ind]**2
                nvec0[ind] += 10**(2*equads[ii][ct]) * np.ones(np.sum(ind))

            # get the basis
            bflags = psr.backend_flags
            Umats = []
            for flag in np.unique(bflags):
                mask = bflags == flag
                Umats.append(utils.create_quantization_matrix(
                    psr.toas[mask])[0])
            nepoch = sum(U.shape[1] for U in Umats)
            U = np.zeros((len(psr.toas), nepoch))
            jvec = np.zeros(nepoch)
            netot = 0
            for ct, flag in enumerate(np.unique(bflags)):
                mask = bflags == flag
                nn = Umats[ct].shape[1]
                U[mask, netot:nn+netot] = Umats[ct]
                jvec[netot:nn+netot] = 10**(2*ecorrs[ii][ct])
                netot += nn

            # get covariance matrix
            cov = np.diag(nvec0) + np.dot(U*jvec[None, :], U.T)
            cf = sl.cho_factor(cov)
            logdet = np.sum(2*np.log(np.diag(cf[0])))
            cfs.append(cf)
            logdets.append(logdet)

            F, f2 = utils.createfourierdesignmatrix_red(psr.toas,
                                                        nmodes=20,
                                                        Tspan=Tspan)
            Mmat = psr.Mmat.copy()
            norm = np.sqrt(np.sum(Mmat**2, axis=0))
            Mmat /= norm
            T = np.hstack((F, Mmat))
            Ts.append(T)
            phi = utils.powerlaw(f2, log10_A=log10_A[ii],
                                 gamma=gamma[ii]) * f2[0]
            if inc_corr:
                phigw = utils.powerlaw(f2, log10_A=GW_log10_A,
                                       gamma=GW_gamma) * f2[0]
            else:
                phigw = np.zeros(40)
            phis.append(np.concatenate(
                (phi+phigw, np.ones(Mmat.shape[1])*1e40)))

        # manually compute loglike
        loglike = 0
        TNrs, TNTs = [], []
        for ct, psr in enumerate(psrs):
            TNrs.append(np.dot(Ts[ct].T, sl.cho_solve(cfs[ct], psr.residuals)))
            TNTs.append(np.dot(Ts[ct].T, sl.cho_solve(cfs[ct], Ts[ct])))
            loglike += -0.5 * (np.dot(psr.residuals, sl.cho_solve(
                cfs[ct], psr.residuals)) + logdets[ct])

        TNr = np.concatenate(TNrs)
        phi = np.diag(np.concatenate(phis))

        if inc_corr:
            hd = utils.hd_orf(psrs[0].pos, psrs[1].pos)
            phi[len(phis[0]):len(phis[0])+40, :40] = np.diag(phigw * hd)
            phi[:40, len(phis[0]):len(phis[0])+40] = np.diag(phigw * hd)

        cf = sl.cho_factor(phi)
        phiinv = sl.cho_solve(cf, np.eye(phi.shape[0]))
        logdetphi = np.sum(2*np.log(np.diag(cf[0])))
        Sigma = sl.block_diag(*TNTs) + phiinv

        cf = sl.cho_factor(Sigma)
        expval = sl.cho_solve(cf, TNr)
        logdetsigma = np.sum(2*np.log(np.diag(cf[0])))

        loglike -= 0.5 * (logdetphi + logdetsigma)
        loglike += 0.5 * np.dot(TNr, expval)

        method = ['partition', 'sparse', 'cliques']
        for mth in method:
            eloglike = pta.get_lnlikelihood(params, phiinv_method=mth)
            msg = 'Incorrect like for npsr={}, phiinv={}'.format(npsrs, mth)
            assert np.allclose(eloglike, loglike), msg

    def test_like_nocorr(self):
        """Test likelihood with no spatial correlations."""
        self.compute_like(npsrs=1)
        self.compute_like(npsrs=2)

    def test_like_corr(self):
        """Test likelihood with spatial correlations."""
        self.compute_like(npsrs=2, inc_corr=True)

    def test_compare_ecorr_likelihood(self):
        """Compare basis and kernel ecorr methods."""

        selection = Selection(selections.nanograv_backends)
        ef = white_signals.MeasurementNoise()
        ec = white_signals.EcorrKernelNoise(selection=selection)
        ec2 = gp_signals.EcorrBasisModel(selection=selection)
        tm = gp_signals.TimingModel()
        m = ef + ec + tm
        m2 = ef + ec2 + tm

        pta1 = signal_base.PTA([m(p) for p in self.psrs])
        pta2 = signal_base.PTA([m2(p) for p in self.psrs])

        params = {p.name: p.sample()[0] for p in pta1.params}

        msg = 'Likelihood mismatch between ECORR methods'
        l1 = pta1.get_lnlikelihood(params)
        l2 = pta2.get_lnlikelihood(params)
        assert np.allclose(l1, l2), msg
