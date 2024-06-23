import gputils
import cupy as cp

from . import direct_sht_pybind11


def _check_lmax_mmax(lmax, mmax):
    if lmax < 0:
        raise RuntimeError()
    if (mmax < 0) or (mmax > lmax):
        raise RuntimeError()

    
def alm_getsize(lmax, mmax=None):
    if mmax is None:
        mmax = lmax

    _check_lmax_mmax(lmax, mmax)
    return ((mmax+1) * (2*lmax-mmax+2)) // 2


def direct_sht(theta, phi, wt, lmax, mmax=None, dest=None):
    """
    Launches a direct SHT on the current cupy device/stream.

    The length of the output array (1-dimensional, dtype complex) is given by alm_getsize(lmax, mmax). 
    The 'dest' argument can be used to pass a caller-allocated output array. 
    If dest is None, then a new output array will be allocated (with cupy.empty()).
    """
    
    if mmax is None:
        mmax = lmax
    if dest is None:
        nalm = alm_getsize(lmax, mmax)
        dest = cp.empty(nalm, dtype=complex)

    stream_ptr = cp.cuda.get_current_stream().ptr
    direct_sht_pybind11._launch_direct_sht(dest, theta, phi, wt, lmax, mmax, stream_ptr)
    
    return dest
