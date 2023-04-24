import numpy as np
from . import cython_fastshermanmorrison as cfsm
from enterprise.signals import signal_base

class FastShermanMorrison(signal_base.ShermanMorrison):
    """Custom container class for Fast-Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        self._uinds = np.vstack([[slc.start, slc.stop] for slc in slices])
        super().__init__(jvec, slices, nvec=nvec)

    def __add__(self, other):
        nvec = self._nvec + other
        return FastShermanMorrison(self._jvec, self._slices, nvec)

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""
        return cfsm.cython_block_shermor_0D(x, self._nvec, self._jvec, self._uinds)

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """
        logJdet, yNx = cfsm.cython_block_shermor_1D1(x, y, self._nvec, self._jvec, self._uinds)
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """
        logJdet, ZNX = cfsm.cython_blas_block_shermor_2D_asymm(Z, X, self._nvec, self._jvec, self._uinds)
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logJdet, xNx = cfsm.cython_block_shermor_1D(np.zeros_like(self._nvec), self._nvec, self._jvec, self._uinds)
        return logJdet

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError("FastShermanMorrison does not implement _solve_D2")
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret

