#ifndef _DIRECT_SHT_HPP
#define _DIRECT_SHT_HPP

#include <gputils/Array.hpp>
#include <cassert>

namespace direct_sht {
#if 0
}  // pacify editor    
#endif


// -------------------------------------------------------------------------------------------------
//
// launch_points2alm() version 1: bare pointer interface.
//
//  - All pointers should point to GPU memory (not host memory).
//
//  - out_alm should be a pointer to the output array, in healpy ordering:
//
//      a_{00} a_{10} ... a_{lmax,0}
//      a_{11} a_{21} ... a_{lmax,1}
//               ...
//      a_{mmax,mmax} ... a_{lmax,mmax}
//
//  - in_theta, in_phi, in_wt should be 1-d arrays of length 'nin'.
//
// FIXME: for now, 'lmax' must be <= 2559 in double precision, 5631 in single precision.
// (Can easily be increased by a factor ~3, increasing further is doable but nontrivial.)
//
// Instatiated for T=float and T=double.


template<typename T>
extern void launch_points2alm(
    std::complex<T> *out_alm,
    const T *in_theta,
    const T *in_phi,
    const T *in_wt,
    int lmax,
    int mmax,
    long nin,
    cudaStream_t stream = nullptr
);


// -------------------------------------------------------------------------------------------------
//
// launch_points2alm() version 2: interface using gputils::Array instead of bare pointers.


template<typename T>
extern void launch_points2alm(
    gputils::Array<std::complex<T>> &out_alm,
    const gputils::Array<T> &in_theta,
    const gputils::Array<T> &in_phi,
    const gputils::Array<T> &in_wt,
    int lmax,
    int mmax,
    cudaStream_t stream = nullptr
);


// -------------------------------------------------------------------------------------------------
//
// alm_offset(), alm_nelts()


// Returns pointer offset at l=0 and specified m, in "units" sizeof(T).
__host__ __device__ inline
int alm_real_offset(int lmax, int m)
{
    int n = (lmax+1) * m;
    return (n << 1) - m*(m-1);
}

// Returns pointer offset at l=0 and specified m, in "units" sizeof(complex<T>).
__host__ inline
int alm_complex_offset(int lmax, int m)
{
    int n = (lmax+1) * m;
    return n - ((m*(m-1)) >> 1);
}


// Returns size of alm array, in "units" sizeof(complex<T>).
__host__ inline
int alm_complex_nelts(int lmax, int mmax)
{
    assert(mmax <= lmax);
    return alm_complex_offset(lmax, mmax+1);  // okay if lmax==mmax
}


// -------------------------------------------------------------------------------------------------
//
// Slow, single-threaded, reference SHT for testing


template<typename T>
extern gputils::Array<std::complex<T>> reference_points2alm(
    const gputils::Array<T> &theta_arr,
    const gputils::Array<T> &phi_arr,
    const gputils::Array<T> &wt_arr,
    int lmax,
    int mmax
);


}  // namespace direct_sht

#endif // _DIRECT_SHT_HPP
