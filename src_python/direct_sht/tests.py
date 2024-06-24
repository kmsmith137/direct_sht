import time
import cupy as cp
import numpy as np

from . import alm_getsize, points2alm_gpu, points2alm_host


def alm_norm(alm, lmax, mmax):
    import healpy
    cl = healpy.sphtfunc.alm2cl(alm, lmax=lmax, mmax=mmax)
    l = np.arange(len(cl))
    return np.sum(cl * (2*l+1))**0.5


def compare_to_healpy(nside, lmax, mmax=None, ngpu=None):
    import healpy
    print(f'direct_sht.compare_to_healpy(): {nside=}, {lmax=}, {mmax=}, {ngpu=}')
    
    npix = 12 * nside**2
    m = np.random.standard_normal(size=npix)
    alm_cpu = healpy.sphtfunc.map2alm(m, lmax=lmax, mmax=mmax, iter=0, use_weights=False, use_pixel_weights=False)
    alm_cpu *= (npix/(4*np.pi))

    theta, phi = healpy.pixelfunc.pix2ang(nside, np.arange(npix))

    # points2alm_host() does the following:
    #   - distributes input arrays (theta,phi,m) from CPU to GPUs,
    #   - computes direct SHT on GPUs
    #   - copies destination arays (alm) from GPUs to CPU, and returns the sum
    
    alm_gpu = points2alm_host(theta, phi, wt=m, lmax=lmax, mmax=mmax, noisy=True)
    
    alm_diff = alm_cpu - alm_gpu
    epsilon = alm_norm(alm_diff,lmax,mmax) / alm_norm(alm_cpu,lmax,mmax)
    print(f'    direct_sht.compare_to_healpy(): {epsilon=}')
    assert epsilon < 1.0e-11
    

def multi_compare_to_healpy(ngpu=None):
    for nside in [ 64, 128, 256, 512, 1024 ]:
        lmax = 2 * nside
        mmax = np.random.randint(nside, 2*nside+1) if (nside < 1024) else None
        compare_to_healpy(nside, lmax, mmax, ngpu)
