#include "../include/direct_sht.hpp"

#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/string_utils.hpp>  // ksgpu::type_name<T> ()
#include <ksgpu/xassert.hpp>

#include <sstream>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

namespace direct_sht {
#if 0
}  // pacify compiler
#endif


// -------------------------------------------------------------------------------------------------
//
// The following notation is used throughout the kernel.
//
//  - K = log2(W), where W is the number of warps per threadblock
//  - J = log2(U), where U is the "unrolling" factor (number of l-values per inner loop)
//  - a is a flattened (l, ReIm) index, i.e. (a0 a1 a2 ...) <-> (ReIm l0 l1 ...)
//  - st is a length-32 spectator index (st0, ..., st31) that we will reduce (sum) over.
//  - sw is a length-W spectator index (sw0, ..., sw_W) that we will reduce (sum) over.
//  - lm refers to the quantity (l-m).
//
// Shared memory layout:
//
//   // Temp buffer for reducing over sw index.
//   T shmem1[W][33]     first index is (sw), second index is (a % 32).
//
//   // Persistent buffer for accumulating alms.
//   T shmem2[M*W][33]   first index is (a // 32), second index is (a % 32).
//
// where M = ceil((lm_max+1)/(16*W)) in order to have enough shared memory for 2*(lm_max+1) elts.


// -------------------------------------------------------------------------------------------------
//
// Compile-time functions.


// Frequently used in static_assert().
constexpr __host__ __device__ bool constexpr_is_pow2(int n)
{
    return (n >= 1) && ((n & (n-1)) == 0);
}


constexpr __host__ __device__ int constexpr_ilog2(int n)
{
    // static_assert() is not allowed in constexpr-functions, so
    // caller should call static_assert(constexpr_is_pow2(n));
    
    return (n > 1) ? (constexpr_ilog2(n/2)+1) : 0;
}


// -------------------------------------------------------------------------------------------------
//
// Some boilerplate, used to support T=float and T=double with the same C++ template.


template<typename T> struct dtype {};


template<> struct dtype<float>
{
    static constexpr float eps = 1.0e-20;
    static constexpr float ieps = 1.0e20;
    static constexpr float sqrt_eps = 1.0e-10;
    static constexpr float sqrt_ieps = 1.0e10;
    static constexpr float log2_eps = -66.43856189774725;
    static constexpr float rec_log2_eps = -0.015051499783199059;
    
    static __device__ float *get_shmem() { extern __shared__ float shmem_f[]; return shmem_f; }
    
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
    static __device__ float xsqrt(float x) { return sqrtf(x); }
    static __device__ float xlog2(float x) { return log2f(x); }
    static __device__ float xexp2(float x) { return exp2f(x); }
    static __device__ void xsincos(float x, float *sptr, float *cptr) { sincosf(x, sptr, cptr); }
};


template<> struct dtype<double>
{
    static constexpr double eps = 1.0e-200;
    static constexpr double ieps = 1.0e200;
    static constexpr double sqrt_eps = 1.0e-100;
    static constexpr double sqrt_ieps = 1.0e100;
    static constexpr double log2_eps = -664.38561897747251805413;
    static constexpr double rec_log2_eps = -0.00150514997831990589;
    
    static __device__ double *get_shmem() { extern __shared__ double shmem_d[]; return shmem_d; }

    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    static __device__ double xsqrt(double x) { return sqrt(x); }
    static __device__ double xlog2(double x) { return log2(x); }
    static __device__ double xexp2(double x) { return exp2(x); }
    static __device__ void xsincos(double x, double *sptr, double *cptr) { sincos(x, sptr, cptr); }
};


// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ T init_Am(int m)
{
    constexpr uint ALL_LANES = 0xffffffffU;
    constexpr T one_over_4pi = 0.079577471545947667884;
    
    // We speed up the computation a bit by parallelizing over threads in the warp.
    // FIXME could make it slightly faster by using all warps, and going through shared memory.
    
    int laneId = threadIdx.x & 0x1f;
    int imax = (m + 31) & ~0x1f;
    T A = T(1);

    for (int i = laneId+1; i <= imax; i += 32) {
	T t = A + A/(2*i);
	A = (i <= m) ? t : A;
    }

    A *= __shfl_sync(ALL_LANES, A, threadIdx.x ^ 0x1);
    A *= __shfl_sync(ALL_LANES, A, threadIdx.x ^ 0x2);
    A *= __shfl_sync(ALL_LANES, A, threadIdx.x ^ 0x4);
    A *= __shfl_sync(ALL_LANES, A, threadIdx.x ^ 0x8);
    A *= __shfl_sync(ALL_LANES, A, threadIdx.x ^ 0x10);
    
    A = dtype<T>::xsqrt(one_over_4pi * A);
    A = (m & 1) ? (-A) : A;

    return A;
}


template<typename T>
__device__ void init_wymm(T &wy, int &wynorm, T s, T w, int m)
{
    // Computes (w * s^m), represented as (to avoid overflows):
    //    (w * s^m) = wy * epsilon**wynorm
    
    if (m == 0) {
	wy = w;
	wynorm = 0;
	return;
    }

    bool wpos = (w > T(0));
    wy = wpos ? T(1) : T(-1);
    w = wpos ? w : (-w);

    bool nonzero = (w > T(0)) && (s > T(0));
    wy = nonzero ? wy : T(0);
    w = nonzero ? w : T(1);
    s = nonzero ? s : T(1);

    T log2_wy = dtype<T>::xlog2(w) + m*dtype<T>::xlog2(s);
    wynorm = int(log2_wy * dtype<T>::rec_log2_eps);
    wynorm = (wynorm > 0) ? wynorm : 0;
    wy *= dtype<T>::xexp2(log2_wy - wynorm * dtype<T>::log2_eps);
}


// -------------------------------------------------------------------------------------------------
//
// Shared memory layout:
//
//   // Temp buffer for reducing over sw index.
//   T shmem1[W][33]     first index is (sw), second index is (a % 32).
//
//   // Persistent buffer for accumulating alms.
//   T shmem2[M*W][33]   first index is (a // 32), second index is (a % 32).


template<typename T, int W>
__host__ int shmem_nbytes(int lmax)
{
    static_assert(constexpr_is_pow2(W));
    constexpr int K = constexpr_ilog2(W);
    
    // M = ceil((lmax+1) / (16*W))
    int M = (lmax >> (K+4)) + 1;

    return (M+1) * W * 33 * sizeof(T);
}


template<typename T, int W>
int shmem_lmax(int max_nbytes)
{
    int Mmax = int(max_nbytes / (33*W*sizeof(T))) - 1;
    int lmax = (Mmax * 16*W) - 1;
    return lmax;
}


template<typename T, int W>
__device__ void initialize_shmem(int lm_max)
{
    constexpr int K = constexpr_ilog2(W);
    T *sp = dtype<T>::get_shmem();

    // Skip shmem1[W][33] (no need to initialize)
    sp += 33*W;

    // Advance by (33*warpId + laneId).
    sp += threadIdx.x + (threadIdx.x >> 5);
    
    // M = ceil((lm_max+1) / (16*W))
    int M = (lm_max >> (K+4)) + 1;

    // Zero shmem2[M*W][33]
    for (int m = 0; m < M; m++)
	sp[m*33*W] = 0;
    
    // Extremely paranoid but that's okay.
    __syncthreads();
}


template<typename T, int W>
__device__ T shmem1_transpose(T x)
{
    // Input:
    //
    //   t0 t1 t2 t3 t4 <-> a0 a1 a2 a3 a4
    //   w0 ... w_{K-1} <-> sw0 ... sw_{K-1}
    //
    // Output:
    //
    //   t0 t1 t2 t3 t4 <-> aK ... a4 sw0 ... sw_{K-1}
    //   w0 ... w_{K-1} <-> a0 ... a_{K-1}
    
    constexpr int K = constexpr_ilog2(W);
    T *sp = dtype<T>::get_shmem();

    int t = threadIdx.x;                   // t = 32*warpId + laneId
    int u = ((t & 0x1f) << K) | (t >> 5);  // u = W*laneId + warpId

    // Input/output shared memory addresses
    int s_in = t + (t >> 5);    // equivalent to 33*warpId + laneId
    int s_out = u + (u >> 5);

    // FIXME use barrier arrive/wait instead of double __syncthreads().
    
    sp[s_in] = x;
    __syncthreads();
    
    T ret = sp[s_out];
    __syncthreads();

    return ret;
}


template<typename T, int W>
__device__ void shmem2_add(T clm, int lm)
{
    // Input:
    //
    //   t0 t1 t2 t3 t4 <-> aK ... a_{K+4}
    //   w0 ... w_{K-1} <-> a0 ... a_{K-1}
    
    constexpr int K = constexpr_ilog2(W);
    T *sp = dtype<T>::get_shmem();

    // Skip shmem1[W][3]]
    sp += 33*W;

    int t = threadIdx.x;                   // t = 32*warpId + laneId
    int u = ((t & 0x1f) << K) | (t >> 5);  // u = W*laneId + warpId
    
    u += ((lm << 1) & ~(32*W-1));    
    sp[u + (u >> 5)] += clm;
    // No __syncthreads() needed.
}


template<typename T, int W>
__device__ void shmem2_to_global(T *out_alm, int lmax, int m)
{
    int n0 = 2 * (lmax - m + 1);
    int npad = (n0+31) & ~31;   // round up to multiple of 32

    // After this shift, 'out_alm' points to an array of length n0.
    out_alm += alm_real_offset(lmax, m);

    T *sp = dtype<T>::get_shmem();
    sp += 33*W;  // skip shmem1[W][33]

    // Important!
    __syncthreads();

    // No warp divergence in outer loop
    for (int i = threadIdx.x; i < npad; i += 32*W) {
	T x = sp[i + (i >> 5)];
	bool access_ok = (i < n0);

	// Can get warp divergence, but that's okay since kernel is about to exit anyway.
	if (access_ok)
	    out_alm[i] = x;
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ T warp_reduce(T x0, T x1, int bit)
{
    constexpr uint ALL_LANES = 0xffffffffU;
    
    // Input:
    //   r <-> i
    //   t0 t1 t2 t3 t4 <-> j0 ... s ... j4
    //
    // where 's' is an index that we want to reduce over.
    //
    // Output:
    //   t0 t1 t2 t3 t4 <-> j0 ... i ... j4
    //
    // The 'bit' argument should be 0x1, 0x2, 0x4, etc.
     
    T src = (threadIdx.x & bit) ? x0 : x1;
    T dst = (threadIdx.x & bit) ? x1 : x0;
    return dst + __shfl_sync(ALL_LANES, src, threadIdx.x ^ bit);
}


template<typename T, int U, int R>
struct advance_sht
{
    // Generic advance_sht -- works for U >= 2
    static_assert((U==2) || (U==4) || (U==8) || (U==16));

    // The meanings of the arguments (lm, ctheta, ..., erec_wide) are explained in the main kernel (points2alm_kernel() below).
    // Advances Ylm recursion by U steps (in l), and returns alm with the following register mapping (see top of file for notation):
    // 	 t0 t1 t2 t3 t4 <-> a0 .... aJ st_{J+1} ... st4
    
    static __device__ __forceinline__ T advance(int lm, T ctheta[R], T cmphi[R], T smphi[R], T wylm[R], T weylm1[R], int wynorm[R], T ewide, T erec_wide)
    {
	// FIXME experiment with alternating registers
	T alm = advance_sht<T,(U/2),R>::advance(lm, ctheta, cmphi, smphi, wylm, weylm1, wynorm, ewide, erec_wide);
	T blm = advance_sht<T,(U/2),R>::advance(lm+U/2, ctheta, cmphi, smphi, wylm, weylm1, wynorm, ewide, erec_wide);
	return warp_reduce(alm, blm, U);
    }
};


template<typename T, int R>
struct advance_sht<T,1,R>
{
    static __device__ __forceinline__ T advance(int lm, T ctheta[R], T cmphi[R], T smphi[R], T wylm[R], T weylm1[R], int wynorm[R], T ewide, T erec_wide)
    {
	constexpr uint ALL_LANES = 0xffffffffU;
	
	T alm_re = 0;
	T alm_im = 0;
	
	T e = __shfl_sync(ALL_LANES, ewide, lm);         // e_{l+1,m}
	T erec = __shfl_sync(ALL_LANES, erec_wide, lm);  // e_{l+1,m}^{-1}

	#pragma unroll
	for (int r = 0; r < R; r++) {
	    T wy = wynorm[r] ? T(0) : wylm[r];
	    alm_re += wy * cmphi[r];
	    alm_im -= wy * smphi[r];  // Note minus sign here (since map->alm transform is defined with Ylm^*)
	    
	    T u = ctheta[r]*wylm[r] - weylm1[r];
	    weylm1[r] = e * wylm[r];
	    wylm[r] = erec * u;
	}
	
	return warp_reduce(alm_re, alm_im, 0x1);
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T, int U, int R, int W, int B>
__global__ void __launch_bounds__(32*W, B)
points2alm_kernel(T *out_alm, const T *in_theta, const T *in_phi, const T *in_wt, int lmax, uint nin)
{
    // Template arguments:
    //   T = float or double.
    //   U = unrolling factor for l-loop.
    //   R = number of (theta,phi) points held in registers on a single thread.
    //   W = number of warps per threadblock (currently 16, see launch_points2alm() below).
    //   B = number of threadblocks per SM (currently 1, see launch_points2alm() below).
    //
    // Grid layout: One threadblock per value of m.
    // The kernel processes points in "outer" blocks of length (32*R*W).
    
    constexpr uint ALL_LANES = 0xffffffffU;
    constexpr int K = constexpr_ilog2(W);
    static_assert(constexpr_is_pow2(W));
    
#if 0
    assert(blockDim.x == 32*W);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
#endif

    const int m = blockIdx.x;
    const int lm_max = (lmax-m) & ~(U-1);
    const T Am = init_Am<T> (m);

    // FIXME use rounded-up lm_max instead of lmax here
    initialize_shmem<T,W> (lmax);
    
    static constexpr uint BS = 32*R*W;  // outer block size
    static constexpr uint BS1 = BS-1;
    uint nin_pad = (nin + BS1) & ~BS1;  // nin, rounded up to multiple of outer block size

    for (uint i_in = threadIdx.x; i_in < nin_pad; i_in += BS) {
	
	// In the outer loop, we compute all alms for (32*R*W) points.
	//
	// Note: our notation for the Ylm recursion is
	//  e_{l+1,m} Y_{l+1,m} = (cos theta) Y_{lm} - e_{lm} Y_{l-1,m}
	//  where e_{lm} = sqrt((l^2-m^2) / (4l^2-1))
	//
	// Note: we deal with underflows by representing:
	//   (wt * Y_{lm}) = wylm[r] * epsilon**wynorm[r]
	//   (wt * e_{lm} * Y_{l-1,m}) = weylm1[r] * epsilon**wynorm[r]
	
	// Per-point data (kept in persistent registers)
	T ctheta[R];    // cos(theta)
	T cmphi[R];     // cos(m*phi)
	T smphi[R];     // sin(m*phi)
	T wylm[R];      // wt * Y_{lm} without e^{i*m*phi} factor
	T weylm1[R];    // wt * e_{lm} Y_{l-1,m} without e^{i*m*phi} factor
	int wynorm[R];  // normalization (in powers of dtype<T>::epsilon)

	// Initialize per-point data.
	#pragma unroll
	for (uint r = 0; r < R; r++) {
	    uint spad = i_in + (32U*W) * r;
	    uint s = (spad < nin) ? spad : (nin-1);
	    
	    T theta = in_theta[s];
	    T phi = in_phi[s];
	    T wt = in_wt[s];
	    
	    wt = (spad < nin) ? wt : 0;

	    T stheta;
	    dtype<T>::xsincos(theta, &stheta, &ctheta[r]);
	    dtype<T>::xsincos(m*phi, &smphi[r], &cmphi[r]);

	    init_wymm(wylm[r], wynorm[r], stheta, Am*wt, m);   // note (Am*wt) here
	    weylm1[r] = T(0);
	}

	T ewide = 0;      // initialized in first iteration of loop
	T erec_wide = 0;  // initialized in first iteration of loop
	T blm = 0;
	T clm = 0;
	
	for (int lm = 0; lm <= lm_max; lm += U) {
	    if ((lm & 0x1f) == 0) {
		// ewide = epsilon_{lm}         at l = m + (lm & ~0x1f) + (laneId) + 1
		// erec_wide = 1/epsilon_{lm}   at l = m + (lm & ~0x1f) + (laneId) + 1

		int l = m + lm + (threadIdx.x & 0x1f) + 1;   // value of l on this thread
		T num = l*l - m*m;
		T den = 4*l*l - 1;
		ewide = dtype<T>::xsqrt(num/den);
		erec_wide = T(1) / ewide;
	    }

	    T alm = advance_sht<T,U,R>::advance(lm, ctheta, cmphi, smphi, wylm, weylm1, wynorm, ewide, erec_wide);

	    #pragma unroll
	    for (int r = 0; r < R; r++) {
		bool renormalize = (wynorm[r] > 0) && ((wylm[r] > 1.0) || (wylm[r] < -1.0));
		T v = renormalize ? dtype<T>::eps : T(1);
		wylm[r] *= v;
		weylm1[r] *= v;
		wynorm[r] = renormalize ? (wynorm[r]-1) : wynorm[r];
	    }
	    
	    // At this point, the alm register mapping is (see top of file for notation):
	    //
	    //   t0 t1 t2 t3 t4 <-> a0 .... aJ st_{J+1} ... st4
	    //   w0 ... w_{K-1} <-> sw0 ... sw_{K-1}
	    //
	    // Reduce over the indices st_{J+1} ... st4, and absorb into (2U) lanes of blm.

	    if constexpr (U <= 1)
		alm += __shfl_sync(ALL_LANES, alm, threadIdx.x ^ 0x2);
	    if constexpr (U <= 2)
		alm += __shfl_sync(ALL_LANES, alm, threadIdx.x ^ 0x4);
	    if constexpr (U <= 4)
		alm += __shfl_sync(ALL_LANES, alm, threadIdx.x ^ 0x8);
	    if constexpr (U <= 8)
		alm += __shfl_sync(ALL_LANES, alm, threadIdx.x ^ 0x10);

	    // Equivalent to (laneId / (2*U)) != ((lm % 16) / U)
	    int keep = ((threadIdx.x >> 1) ^ lm) & (16-U);
	    blm = keep ? blm : alm;

	    // Code after this point is independent of U (the loop-unrolling factor).
	    // After 16 iterations of the 'lm' loop, the blm register mapping is:
	    //
	    //   t0 t1 t2 t3 t4 <-> a0 a1 a2 a3 a4
	    //   w0 ... w_{K-1} <-> sw0 ... sw_{K-1}
	    //
	    // We process the blm array every 16 iterations, and in the last iteration.
	    
	    if ((lm < lm_max) && (~lm & (16-U)))
		continue;

	    // Transpose in shared memory, obtaining blm register mapping:
	    //
	    //   t0 t1 t2 t3 t4 <-> aK ... a4 sw0 ... sw_{K-1}
	    //   w0 ... w_{K-1} <-> a0 ... a_{K-1}
	    
	    blm = shmem1_transpose<T,W> (blm);

	    // Reduce over sw0, ..., sw_{K-1} and absorb the result into (32/W) lanes of the 'clm' array.
	    // This step can be done more efficiently, but I doubt it's worth the extra registers.
	    
	    if constexpr (W > 16)
		blm += __shfl_sync(ALL_LANES, blm, threadIdx.x ^ 0x1);
	    if constexpr (W > 8)
		blm += __shfl_sync(ALL_LANES, blm, threadIdx.x ^ 0x2);
	    if constexpr (W > 4)
		blm += __shfl_sync(ALL_LANES, blm, threadIdx.x ^ 0x4);
	    if constexpr (W > 2)
		blm += __shfl_sync(ALL_LANES, blm, threadIdx.x ^ 0x8);
	    if constexpr (W > 1)
		blm += __shfl_sync(ALL_LANES, blm, threadIdx.x ^ 0x10);

	    // Upper K bits of threadIdx should agree with bits [4:4+K] of lm.
	    int keep2 = ((threadIdx.x >> (5-K)) ^ (lm >> 4)) & (W-1);
	    clm = keep2 ? clm : blm;
	    blm = 0;  // Reset blm, after it gets absorbed into clm.
	    
	    // After (16*W) iterations of the 'lm' loop, the clm register mapping is:
	    //
	    //   t0 t1 t2 t3 t4 <-> aK ... a_{K+4}
	    //   w0 ... w_{K-1} <-> a0 ... a_{K-1}
	    //
	    // We process the clm array every (16*W) iterations, and in the last iteration.

	    if ((lm < lm_max) && (~lm & (16*W-U)))
		continue;
	    
	    shmem2_add<T,W> (clm, lm);
	    clm = 0;  // Reset clm, after it gets absorbed into shared memory.
	}
    }

    // All done -- update global memory.
    shmem2_to_global<T,W> (out_alm, lmax, m);
}


// -------------------------------------------------------------------------------------------------
//
// Template instantiations.


template<typename T>
void launch_points2alm(complex<T> *out_alm, const T *in_theta, const T *in_phi, const T *in_wt, int lmax, int mmax, long npoints, cudaStream_t stream)
{
    // FIXME placeholder values for testing
    constexpr int U = 4;
    constexpr int R = 4;
    constexpr int W = 16;
    constexpr int B = 1;
    constexpr int BS = 32*R*W;   // kernel block size
    constexpr long npoints_max = (1L << 32) - BS;

    if (npoints <= 0)
	throw runtime_error("launch_points2alm(): 'npoints' argument must be > 0");
    if (npoints > npoints_max)
	throw runtime_error("launch_points2alm(): 'npoints' argument must be <= " + std::to_string(npoints_max));
    
    xassert_msg(mmax >= 0, "direct_sht.gpu_points2alm() was called with mmax < 0");
    xassert_msg(lmax >= mmax, "direct_sht.gpu_points2alm() was called with lmax < mmax");
    xassert_msg(npoints > 0, "direct_sht.gpu_points2alm() was called with npoints <= 0");
    xassert_msg(npoints <= npoints_max, "direct_sht.gpu_points2alm(): 'npoints' argument must be <= " + std::to_string(npoints_max));

    // FIXME some day, I'd like to write a unit test that tests nin == nin_max.

    // FIXME currently limited to artificially low lmax
    constexpr int max_shmem_nbytes = 48 * 1024;
    int sb = shmem_nbytes<T,W> (lmax);    

    if (sb > max_shmem_nbytes) {
	std::stringstream ss;
	ss << "launch_points2alm(): FIXME: called with lmax="
	   << lmax << ", and we're currently limited to artifically low lmax="
	   << shmem_lmax<T,W> (max_shmem_nbytes) << " for dtype="
	   << ksgpu::type_name<T> ();
	throw std::runtime_error(ss.str());
    }
    
    points2alm_kernel<T,U,R,W,B>
	<<< mmax+1, 32*W, sb, stream >>>
	(reinterpret_cast<T *> (out_alm), in_theta, in_phi, in_wt, lmax, npoints);

    CUDA_PEEK("points2alm_kernel launch");
}


template<typename T>
void launch_points2alm(Array<complex<T>> &out_alm, const Array<T> &in_theta, const Array<T> &in_phi, const Array<T> &in_wt, int lmax, int mmax, cudaStream_t stream)
{
    check_array_arg(out_alm, "direct_sht.gpu_points2alm()", "alm", true);     // on_gpu=true
    check_array_arg(in_theta, "direct_sht.gpu_points2alm()", "theta", true);  // on_gpu=true
    check_array_arg(in_phi, "direct_sht.gpu_points2alm()", "phi", true);      // on_gpu=true
    check_array_arg(in_wt, "direct_sht.gpu_points2alm()", "wt", true);        // on_gpu=true

    xassert_msg(in_theta.size == in_phi.size, "direct_sht.gpu_points2alm() was called with theta,phi arrays of different sizes");
    xassert_msg(in_theta.size == in_wt.size, "direct_sht.gpu_points2alm() was called with theta,wt arrays of different sizes");

    int nalm_expected = alm_complex_nelts(lmax, mmax);
    xassert_msg(in_theta.size == in_wt.size, "direct_sht.gpu_points2alm() was called with wrong-size alm output array");

    // Checks lmax, mmax.
    launch_points2alm<T> (out_alm.data, in_theta.data, in_phi.data, in_wt.data, lmax, mmax, in_theta.size, stream);
}


#define INSTANTIATE(T) \
    template void launch_points2alm(complex<T> *out_alm, const T *in_theta, const T *in_phi, \
				    const T *in_wt, int lmax, int mmax, long nin, cudaStream_t stream); \
    template void launch_points2alm(Array<complex<T>> &out_alm, const Array<T> &in_theta, const Array<T> &in_phi, \
				    const Array<T> &in_wt, int lmax, int mmax, cudaStream_t stream)
				        
INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace direct_sht
