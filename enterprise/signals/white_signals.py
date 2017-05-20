# white_signals.py
"""Contains class factories for white noise signals. White noise signals are
defined as the class of signals that only modifies the white noise matrix `N`.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import scipy.sparse

from enterprise.signals import parameter
import enterprise.signals.signal_base as base
from enterprise.signals import utils
from enterprise.signals import selections
from enterprise.signals.selections import Selection


def MeasurementNoise(efac=parameter.Uniform(0.5,1.5),
                     selection=Selection(selections.no_selection)):
    """Class factory for EFAC type measurement noise."""

    class MeasurementNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'efac'

        def __init__(self, psr):

            sel = selection(psr)
            self._params, self._ndiag = sel('efac', efac, psr.toaerrs**2)

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [self.get(p, params)**2*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return MeasurementNoise


def EquadNoise(log10_equad=parameter.Uniform(-10,-5),
               selection=Selection(selections.no_selection)):
    """Class factory for EQUAD type measurement noise."""

    class EquadNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'equad'

        def __init__(self,psr):

            sel = selection(psr)
            self._params, self._ndiag = sel('log10_equad', log10_equad,
                                            np.ones_like(psr.toaerrs))

        def get_ndiag(self, params):
            ret = base.ndarray_alt(np.sum(
                [10**(2*self.get(p, params))*self._ndiag[p]
                 for p in self._params], axis=0))
            return ret

    return EquadNoise


def EcorrKernelNoise(log10_ecorr=parameter.Uniform(-10, -5),
                     selection=Selection(selections.no_selection)):
    """Class factory for ECORR type noise using Sparse method."""

    class EcorrKernelNoise(base.Signal):
        signal_type = 'white noise'
        signal_name = 'ecorr_sparse'

        def __init__(self, psr):

            # TODO: Add check for proper TOA sorting

            sel = selection(psr)
            self._params, self._masks = sel('log10_ecorr', log10_ecorr)
            Umats = {}
            for key in sorted(self._masks.keys()):
                mask = self._masks[key]
                Umats.update({key: utils.create_quantization_matrix(
                    psr.toas[mask], nmin=1)})
            nepoch = np.sum(U.shape[1] for U in Umats.values())
            self._F = np.zeros((len(psr.toas), nepoch))
            netot = 0
            self._slices = {}
            for key in sorted(self._masks.keys()):
                mask = self._masks[key]
                Umat = Umats[key]
                nn = Umat.shape[1]
                self._F[mask, netot:nn+netot] = Umat
                self._slices.update({key: utils.quant2ind(
                    self._F[:,netot:nn+netot])})
                netot += nn

            # initialize sparse matrix
            self._setup(psr)

        def _setup(self, psr):
            Ns = scipy.sparse.csc_matrix((len(psr.toas), len(psr.toas)))
            for key, slices in self._slices.items():
                for slc in slices:
                    if slc.stop - slc.start > 1:
                        Ns[slc, slc] = 1.0
            self._Ns = base.csc_matrix_alt(Ns)

        def get_ndiag(self, params):
            for p in self._params:
                for slc in self._slices[p]:
                    if slc.stop - slc.start > 1:
                        self._Ns[slc, slc] = 10**(2*self.get(p, params))
            return self._Ns

    return EcorrKernelNoise


def EcorrKernelNoiseSM(log10_ecorr=parameter.Uniform(-10, -5),
                       selection=Selection(selections.no_selection)):
    """Class factory for ECORR type noise using Sherman-Morrison method."""

    BaseClass = EcorrKernelNoise(log10_ecorr=log10_ecorr, selection=selection)

    class EcorrKernelNoiseSM(BaseClass):

        def _setup(self, psr):
            pass

        def get_ndiag(self, params):
            slices = sum([self._slices[key] for key in
                          sorted(self._slices.keys())], [])
            jvec = np.concatenate(
                [np.ones(len(self._slices[key]))*10**(2*self.get(key, params))
                 for key in sorted(self._slices.keys())])

            return base.ShermanMorrison(jvec, slices)

    return EcorrKernelNoiseSM


def EcorrKernelNoiseBlock(log10_ecorr=parameter.Uniform(-10, -5),
                          selection=Selection(selections.no_selection)):
    """Class factory for ECORR type noise using Block method."""

    BaseClass = EcorrKernelNoiseSM(log10_ecorr=log10_ecorr, selection=selection)

    class EcorrKernelNoiseBlock(BaseClass):

        def get_ndiag(self, params):
            slices = sum([self._slices[key] for key in
                          sorted(self._slices.keys())], [])
            jvec = np.concatenate(
                [np.ones(len(self._slices[key]))*10**(2*self.get(key, params))
                 for key in sorted(self._slices.keys())])

            blocks = []
            for jv, slc in zip(jvec, slices):
                nb = slc.stop - slc.start
                blocks.append(np.ones((nb, nb))*jv)

            return base.BlockMatrix(blocks, slices)

    return EcorrKernelNoiseBlock
