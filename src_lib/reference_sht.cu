#include "../include/direct_sht.hpp"

#include <cmath>
#include <iostream>

using namespace std;
using namespace ksgpu;

namespace direct_sht {
#if 0
}  // compiler pacifier
#endif


template<typename T> struct cpu_dtype { };

template<> struct cpu_dtype<float>
{
    static float xsqrt(float x) { return sqrtf(x); }
    static float xexp2(float x) { return exp2f(x); }
    static float xlog2(float x) { return log2f(x); }
    static float xpow(float x, int m) { return powf(x,m); }
    static void xsincos(float x, float *sp, float *cp) { sincosf(x, sp, cp); }
};

template<> struct cpu_dtype<double>
{
    static double xsqrt(double x) { return sqrt(x); }
    static double xexp2(double x) { return exp2(x); }
    static double xlog2(double x) { return log2(x); }
    static double xpow(double x, int m) { return pow(x,m); }
    static void xsincos(double x, double *sp, double *cp) { sincos(x, sp, cp); }
};


// -------------------------------------------------------------------------------------------------


template<typename T>
inline T epsilon(int l, int m)
{
    T num = l*l - m*m;
    T den = 4*l*l - 1;
    return cpu_dtype<T>::xsqrt(num/den);
}


template<typename T>
Array<complex<T>> reference_points2alm(const Array<T> &theta_arr, const Array<T> &phi_arr, const Array<T> &wt_arr, int lmax, int mmax)
{
    const T sqrt_one_over_4pi = cpu_dtype<T>::xsqrt(1.0 / (4*M_PI));
    
    check_array_arg(theta_arr, "direct_sht.reference_points2alm()", "theta", false);  // on_gpu=false
    check_array_arg(phi_arr, "direct_sht.reference_points2alm()", "phi", false);      // on_gpu=false
    check_array_arg(wt_arr, "direct_sht.reference_points2alm()", "wt", false);        // on_gpu=false
    
    xassert_msg(mmax >= 0, "direct_sht.reference_points2alm() was called with mmax < 0");
    xassert_msg(lmax >= mmax, "direct_sht.reference_points2alm() was called with lmax < mmax");
    xassert_msg(theta_arr.size == phi_arr.size, "direct_sht.reference_points2alm() was called with theta,phi arrays of different sizes");
    xassert_msg(theta_arr.size == wt_arr.size, "direct_sht.reference_points2alm() was called with theta,wt arrays of different sizes");

    int nin = theta_arr.size;
    int nalm = alm_complex_nelts(lmax, mmax);
    
    Array<complex<T>> out({nalm}, af_uhost | af_zero);
    complex<T> *out_p = out.data;

    const T *theta_p = theta_arr.data;
    const T *phi_p = phi_arr.data;
    const T *wt_p = wt_arr.data;

    // Extremely slow
    for (int i = 0; i < nin; i++) {
	T Am_abs = sqrt_one_over_4pi;
	T theta = theta_p[i];
	T phi = phi_p[i];
	T wt = wt_p[i];

	T sin_theta, cos_theta;
	cpu_dtype<T>::xsincos(theta, &sin_theta, &cos_theta);
	
	// FIXME assumed below when taking log(sin(theta))
	xassert(sin_theta > 0.0);
	
	for (int m = 0; m <= mmax; m++) {
	    complex<T> *out_mslice = out_p + alm_complex_offset(lmax,m);
	    
	    T sin_mphi, cos_mphi;
	    cpu_dtype<T>::xsincos(m*phi, &sin_mphi, &cos_mphi);
	    complex<T> eimphi = { cos_mphi, -sin_mphi };  // Note minus sign here, since map2alm SHT is defined with Ylm^*

	    // In our reference implementation, we represent (W*Y_{lm}, W*Y_{l-1,m}) by
	    // a triple (alpha, beta, gamma) such that:
	    //
	    //  Y_{lm} = alpha * exp2(gamma)
	    //  Y_{l-1,m} = beta * exp2(gamma)
	    //
	    // where alpha^2 + beta^2 == 1

	    // Note no A_m here
	    T beta = T(0);
	    T alpha = (m & 1) ? T(-1) : T(1);
	    T gamma = cpu_dtype<T>::xlog2(Am_abs) + m * cpu_dtype<T>::xlog2(sin_theta);
	    T el = 0;
	    
	    for (int l = m; l <= lmax; l++) {
		T wy = wt * alpha * cpu_dtype<T>::xexp2(gamma);
		out_mslice[l-m] += wy * eimphi;

		T el_next = epsilon<T> (l+1, m);
		T alpha_next = (cos_theta*alpha - el*beta) / el_next;

		T t = alpha_next*alpha_next + alpha*alpha;
		xassert(t > 0.0);

		T u = cpu_dtype<T>::xsqrt(t);
		beta = alpha / u;
		alpha = alpha_next / u;
		gamma += cpu_dtype<T>::xlog2(u);
		el = el_next;
	    }
	    
	    // Update A_m -> A_{m+1}
	    Am_abs *= cpu_dtype<T>::xsqrt(1.0 + 1.0/(2*m+2));
	}
    }

    return out;
}


#define INSTANTIATE(T) \
    template Array<complex<T>> reference_points2alm(const Array<T> &, const Array<T> &, const Array<T> &, int, int)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace direct_sht
