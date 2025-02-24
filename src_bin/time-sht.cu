#include "../include/direct_sht.hpp"

#include <iostream>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/time_utils.hpp>
#include <ksgpu/string_utils.hpp>

using namespace std;
using namespace ksgpu;
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
	launch_points2alm(alm, theta, phi, wt, lmax, mmax);

    CUDA_CALL(cudaDeviceSynchronize());
    double dt = time_diff(tv0, get_time());
    double tflops = 1.0e-12 * nflops / dt;
	
    cout << "    Elapsed time = " << dt << " seconds" << endl
	 << "    Tflops = " << tflops << endl;
}

	
int main(int argc, char **argv)
{
    cout << "Note: in double precision, this cuda program (time-sht.cu) is equivalent to 'python -m direct_sht time'\n"
	 << "However, I'm keeping the C++ program around since it's currently the only way to time the single-precison transforms."
	 << endl;
    
    time_sht<float> (3*5 * 128 * 1024, 1000, 1000);
    time_sht<double> (3*5 * 128 * 1024, 1000, 1000);
    
    return 0;
}
