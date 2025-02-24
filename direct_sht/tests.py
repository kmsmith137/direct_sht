"""To run all unit tests, do 'python -m direct_sht test'."""

import time
import cupy as cp
import numpy as np

from . import alm_getsize, points2alm_gpu, points2alm_host
from . import direct_sht_pybind11


def alm_norm(alm, lmax, mmax):
    import healpy
    cl = healpy.sphtfunc.alm2cl(alm, lmax=lmax, mmax=mmax)
    l = np.arange(len(cl))
    return np.sum(cl * (2*l+1))**0.5

def alm_compare(alm1, alm2, lmax, mmax):
    return alm_norm(alm1-alm2,lmax,mmax) / alm_norm(alm1,lmax,mmax)


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
    epsilon = alm_compare(alm_cpu, alm_gpu, lmax, mmax)
    print(f'    direct_sht.compare_to_healpy(): {epsilon=}')
    assert epsilon < 1.0e-11
    

def multi_compare_to_healpy(ngpu=None):
    for nside in [ 64, 128, 256, 512, 1024 ]:
        lmax = 2 * nside
        mmax = np.random.randint(nside, 2*nside+1) if (nside < 1024) else None
        compare_to_healpy(nside, lmax, mmax, ngpu)


####################################################################################################


def points2alm_reference(theta, phi, wt, lmax, mmax):
    if mmax is None:
        mmax = lmax
    return direct_sht_pybind11.reference_points2alm(theta, phi, wt, lmax, mmax)
    

def compare_to_reference(nlg, nsm, lmax, mmax, ngpu):
    print(f'direct_sht.compare_to_reference(): {nlg=}, {nsm=}, {lmax=}, {mmax=}, {ngpu=}')
    
    theta_lg = np.random.uniform(0.001*np.pi, 0.999*np.pi, size=nlg)
    phi_lg = np.random.uniform(0, 2*np.pi, size=nlg)
    wt_lg = np.zeros(nlg)

    theta_sm = np.zeros(nsm)
    phi_sm = np.zeros(nsm)
    wt_sm = np.random.uniform(size=nsm)
    
    for ism in range(nsm):
        ilg = np.random.randint(0,nlg)
        theta_sm[ism] = theta_lg[ilg]
        phi_sm[ism] = phi_lg[ilg]
        wt_lg[ilg] += wt_sm[ism]

    alm_gpu = points2alm_host(theta_lg, phi_lg, wt_lg, lmax=lmax, mmax=mmax, ngpu=ngpu, noisy=True)
    alm_cpu = points2alm_reference(theta_sm, phi_sm, wt_sm, lmax=lmax, mmax=mmax)
    epsilon = alm_compare(alm_cpu, alm_gpu, lmax, mmax)
    print(f'    direct_sht.compare_to_reference(): {epsilon=}')
    assert epsilon < 1.0e-11


def multi_compare_to_reference():
    ndev = cp.cuda.runtime.getDeviceCount()
    
    for _ in range(100):
        lmax = np.random.randint(0, 1001)
        mmax = np.random.randint(0, lmax+1) if (np.random.uniform() < 0.9) else None
        nsm = np.random.randint(30, 100)
        nlg = np.random.randint(nsm, 16*1024)
        ngpu = np.random.randint(1, ndev+1)
        
        compare_to_reference(nlg, nsm, lmax, mmax, ngpu)
