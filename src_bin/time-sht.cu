#include "../include/direct_sht.hpp"

#include <cassert>
#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace direct_sht;


// -------------------------------------------------------------------------------------------------


template<typename T>
static void time_sht(long nin, int lmax, int mmax, int niter=1)
{
    int nalm = alm_complex_nelts(lmax, mmax);
    double nflops = 10.0 * double(niter) * double(nin) * double(nalm);  // 5 FMAs per (Re(alm), Im(alm)) pair
    
    cout << "time_sht<" << type_name<T>() << ">: nin=" << nin << ", lmax=" << lmax
	 << ", mmax=" << mmax << ", niter=" << niter << ", tflops=" << (1.0e-12 * nflops)
	 << endl;

    Array<complex<T>> alm({nalm}, af_gpu | af_zero);
    Array<T> theta({nin}, af_rhost | af_zero);
    Array<T> phi({nin}, af_gpu | af_zero);
    Array<T> wt({nin}, af_gpu | af_zero);

    T *p = theta.data;
    for (long i = 0; i < nin; i++)
	p[i] = T(M_PI/2);

    theta = theta.to_gpu();
    
    CUDA_CALL(cudaDeviceSynchronize());
    struct timeval tv0 = get_time();

    for (int i = 0; i < niter; i++)
	launch_direct_sht(alm, theta, phi, wt, lmax, mmax);

    CUDA_CALL(cudaDeviceSynchronize());
    double dt = time_diff(tv0, get_time());
    double tflops = 1.0e-12 * nflops / dt;
	
    cout << "    Elapsed time = " << dt << " seconds" << endl
	 << "    Tflops = " << tflops << endl;
}

	
int main(int argc, char **argv)
{
    time_sht<float> (3*5 * 128 * 1024, 1000, 1000);
    time_sht<double> (3*5 * 128 * 1024, 1000, 1000);
    return 0;
}
