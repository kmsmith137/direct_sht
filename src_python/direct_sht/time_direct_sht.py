import time
import cupy as cp

from . import direct_sht, alm_getsize


def time_single_gpu(nin, lmax, mmax, niter):
    nalm = alm_getsize(lmax, mmax)
    tflop_count = 1.0e-11 * niter * nin * nalm  # 5 FMAs per (Re(alm), Im(alm)) pair

    print(f'time_direct_sht: running on single GPU')
    print(f'    time_direct_sht: {nin=} {lmax=} {mmax=} {niter=} {tflop_count=}')

    # Allocate arrays on the GPU
    alm = cp.zeros(nalm, dtype=complex)
    theta = cp.zeros(nin, dtype=float)
    phi = cp.zeros(nin, dtype=float)
    wt = cp.zeros(nin, dtype=float)
    
    t0 = time.time()

    # Launch kernel asynchronously on GPU.
    # Note that we use the 'dest' argument.
    for _ in range(niter):
        direct_sht(theta, phi, wt, lmax, mmax, dest=alm)

    # Synchronize
    cp.cuda.Device(0).synchronize()
    
    dt = time.time() - t0
    tflops = tflop_count / dt
    print(f'    time_direct_sht: {dt} seconds, {tflops=}')


def time_multiple_gpus(nin, lmax, mmax, niter):
    ngpu = cp.cuda.runtime.getDeviceCount()
    nalm = alm_getsize(lmax, mmax)
    tflop_count = 1.0e-11 * ngpu * niter * nin * nalm  # note ngpu here
    
    print(f'time_direct_sht: running on multiple GPUs')
    print(f'    time_direct_sht: {nin=} {lmax=} {mmax=} {niter=} {tflop_count=}')

    # Allocate arrays on the GPUs
    alm = [ ]
    theta = [ ]
    phi = [ ]
    wt = [ ]

    for idev in range(ngpu):
        with cp.cuda.Device(idev):
            alm.append(cp.zeros(nalm, dtype=complex))
            theta.append(cp.zeros(nin, dtype=float))
            phi.append(cp.zeros(nin, dtype=float))
            wt.append(cp.zeros(nin, dtype=float))
    
    t0 = time.time()

    # Launch kernel asynchronously on GPU.
    # Note that we use the 'dest' argument.
    for _ in range(niter):
        for idev in range(ngpu):
            with cp.cuda.Device(idev):
                direct_sht(theta[idev], phi[idev], wt[idev], lmax, mmax, dest=alm[idev])

    # Synchronize
    for idev in range(ngpu):
        with cp.cuda.Device(idev):
            cp.cuda.Device(idev).synchronize()
    
    dt = time.time() - t0
    tflops = tflop_count / dt
    print(f'    time_direct_sht: {dt} seconds, {tflops=}')


def time_direct_sht(nin, lmax, mmax=None, niter=1):
    time_single_gpu(nin, lmax, mmax, niter)
        
    if cp.cuda.runtime.getDeviceCount() > 1:
        time_multiple_gpus(nin, lmax, mmax, niter)
    else:
        print('time_direct_sht: this machine has 1 GPU -- skipping multi-GPU timing')
