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
// launch_direct_sht() version 1: bare pointer interface.
//
//  - All pointers should point to GPU memory (not host memory).
//
//  - out_alm should be a pointer to the output array, in the following ordering:
//
//      Re(m=0,l=0) Im(m=0,l=0) ... Re(m=0,l=lmax) Im(m=0,l=lmax)
//      Re(m=1,l=1) Im(m=1,l=1) ... Re(m=1,l=lmax) Im(m=1,l=lmax)
//      Re(m=mmax,l=mmax) Im(m=mmax,l=mmax) ... Re(m=mmax,l=lmax) Im(m=mmax,l=lmax)
//
//  - in_theta, in_phi, in_wt should be 1-d arrays of length 'nin'.
//
// FIXME: for now, 'nin' must be a multiple of 2048. (This would be easy to fix.)
//
// FIXME: for now, 'lmax' must be <= 2559 in double precision, 5631 in single precision.
// (Can easily be increased by a factor ~3, increasing further is doable but nontrivial.)
//
// Instatiated for T=float and T=double.


template<typename T>
extern void launch_direct_sht(
    T *out_alm,
    const T *in_theta,
    const T *in_phi,
    const T *in_wt,
    int lmax,
    int mmax,
    long nin
);


// -------------------------------------------------------------------------------------------------
//
// launch_direct_sht() version 2: interface using gputils::Array instead of bare pointers.


template<typename T>
extern void launch_direct_sht(
    gputils::Array<T> &out_alm,
    const gputils::Array<T> &in_theta,
    const gputils::Array<T> &in_phi,
    const gputils::Array<T> &in_wt,
    int lmax,
    int mmax
);


// -------------------------------------------------------------------------------------------------
//
// alm_offset(), alm_nelts()


// Returns pointer offset at l=m, in "units" sizeof(T), not sizeof(complex<T>)
__host__ __device__ inline
int alm_offset(int lmax, int m)
{
    int n = (lmax+1) * m;
    return (n << 1) - m*(m-1);
}


// Returns size of alm array, in "units" sizeof(T), not sizeof(complex<T>)
__host__ inline
int alm_nelts(int lmax, int mmax)
{
    assert(mmax <= lmax);
    return alm_offset(lmax, mmax+1);  // okay if lmax==mmax
}


// -------------------------------------------------------------------------------------------------
//
// Slow, single-threaded, reference SHT for testing


template<typename T>
extern gputils::Array<T> reference_sht(
    const gputils::Array<T> &theta_arr,
    const gputils::Array<T> &phi_arr,
    const gputils::Array<T> &wt_arr,
    int lmax,
    int mmax
);


}  // namespace direct_sht

#endif // _DIRECT_SHT_HPP