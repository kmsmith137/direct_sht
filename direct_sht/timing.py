"""To time the direct SHT, do 'python -m direct_sht time'."""

import time
import cupy as cp
import numpy as np

from . import alm_getsize, points2alm_gpu, points2alm_host

        
def time_points2alm(npoints_per_gpu, lmax, mmax=None, ngpu=None):
    if ngpu is None:
        # First run: one GPU
        time_points2alm(npoints_per_gpu, lmax, mmax, ngpu=1)

        # Second run: multiple GPUs
        n = cp.cuda.runtime.getDeviceCount()
        if n > 1:
            time_points2alm(npoints_per_gpu, lmax, mmax, ngpu=n)
        
        return
    
    nalm = alm_getsize(lmax, mmax)
    tflop_count = 1.0e-11 * ngpu * npoints_per_gpu * nalm  # note ngpu here
    
    print(f'time_points2alm: running on {ngpu} GPU(s)')
    print(f'    time_points2alm: {npoints_per_gpu=} {lmax=} {mmax=} {tflop_count=}')

    # Allocate arrays on the GPUs
    alm = [ ]
    theta = [ ]
    phi = [ ]
    wt = [ ]

    for idev in range(ngpu):
        with cp.cuda.Device(idev):
            alm.append(cp.zeros(nalm, dtype=complex))
            theta.append(cp.zeros(npoints_per_gpu, dtype=float))
            phi.append(cp.zeros(npoints_per_gpu, dtype=float))
            wt.append(cp.zeros(npoints_per_gpu, dtype=float))
    
    t0 = time.time()

    # Launch kernel asynchronously on GPU.
    # Note that we use the 'dest' argument.
    for idev in range(ngpu):
        with cp.cuda.Device(idev):
            points2alm_gpu(theta[idev], phi[idev], wt[idev], lmax, mmax, dest=alm[idev])

    # Synchronize
    for idev in range(ngpu):
        with cp.cuda.Device(idev):
            cp.cuda.Device(idev).synchronize()
    
    dt = time.time() - t0
    tflops = tflop_count / dt
    print(f'    time_points2alm: {dt} seconds, {tflops=}')

    # Now run through points2alm_host()
    alm = [ ]
    theta = np.zeros(ngpu * npoints_per_gpu)
    phi = np.zeros(ngpu * npoints_per_gpu)
    wt = np.zeros(ngpu * npoints_per_gpu)

    t0 = time.time()
    points2alm_host(theta, phi, wt, lmax, mmax, ngpu, noisy=True)
    
    dt = time.time() - t0
    tflops = tflop_count / dt
    print(f'    called through points2alm_host() wrapper: {dt} seconds, {tflops=}')
