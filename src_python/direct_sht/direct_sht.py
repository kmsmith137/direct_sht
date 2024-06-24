import gputils
import cupy as cp
import numpy as np

from . import direct_sht_pybind11


def _check_lmax_mmax(lmax, mmax):
    if lmax < 0:
        raise RuntimeError("Expected lmax >= 0")
    if (mmax < 0) or (mmax > lmax):
        raise RuntimeError("Expected 0 <= mmax <= lmax")


def _check_host_array(arr, arr_name):
    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"Expected '{arr_name}' to be a numpy array")
    if arr.ndim != 1:
        raise RuntimeError(f"Expected '{arr_name}' to be a 1-dimensional array (shape={arr.shape})")
    
    
def alm_getsize(lmax, mmax=None):
    if mmax is None:
        mmax = lmax

    _check_lmax_mmax(lmax, mmax)
    return ((mmax+1) * (2*lmax-mmax+2)) // 2


def points2alm_gpu(theta, phi, wt, lmax, mmax=None, dest=None):
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


def points2alm_host(theta, phi, wt, lmax, mmax=None, ngpu=None, noisy=False):
    """
    Simple wrapper -- not always the most efficient.
    Uses current cupy stream on each GPU.
    """

    _check_host_array(theta, 'theta')
    _check_host_array(phi, 'phi')
    _check_host_array(wt, 'wt')

    if not (theta.size == phi.size == wt.size):
        raise RuntimeError('Exepected theta, phi, wt arrays to be the same size')

    npoints = theta.size
    nalm = alm_getsize(lmax, mmax)
    
    if ngpu is None:
        # Use at least 1024 points per GPU
        ngpu = min(cp.cuda.runtime.getDeviceCount(), npoints//1024 + 1)

    if noisy:
        print(f'    direct_sht.points2alm_host(): distributing {npoints} points to {ngpu} GPUs')

    delim = [ (i*npoints) // ngpu for i in range(ngpu+1) ]
    alm_gpu = [ ]

    for i in range(ngpu):
        with cp.cuda.Device(i):
            # Copy points from CPU to GPU.
            # FIXME async copies may speed this up.
            lo, hi = delim[i], delim[i+1]
            th_gpu = cp.asarray(theta[lo:hi])
            ph_gpu = cp.asarray(phi[lo:hi])
            wt_gpu = cp.asarray(wt[lo:hi])
            
            # Allocate array on the i-th GPU
            alm = cp.empty(nalm, dtype=complex)
            alm_gpu.append(alm)

            # Asynchronously Launch kernel on the i-th GPU
            points2alm_gpu(th_gpu, ph_gpu, wt_gpu, lmax, mmax, dest=alm)

    # Copy array from GPU0 to CPU (will block until kernel completes)
    alm = cp.asnumpy(alm_gpu[0])

    # Accumulate contributions from GPUs >= 1 (will block until kernels complete)
    for i in range(1, ngpu):
        alm += cp.asnumpy(alm_gpu[i])

    return alm
