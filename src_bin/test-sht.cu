#include "../include/direct_sht.hpp"
#include <ksgpu/rand_utils.hpp>    // rand_uniform()
#include <ksgpu/test_utils.hpp>    // assert_arrays_equal()
#include <ksgpu/string_utils.hpp>  // type_name<T>()

#include <iostream>

using namespace std;
using namespace ksgpu;
using namespace direct_sht;


// FIXME move to ksgpu (with a better name)
template<typename T> struct Eps { };
template<> struct Eps<float> { static constexpr float eps = 1.0e-3; };
template<> struct Eps<double> { static constexpr double eps = 1.0e-13; };

// -------------------------------------------------------------------------------------------------


template<typename T>
Array<T> rand_array(long nelts, T lo, T hi)
{
    xassert(nelts > 0);
    Array<T> ret({nelts}, af_uhost);
    
    for (long i = 0; i < nelts; i++)
	ret.data[i] = ksgpu::rand_uniform(lo, hi);

    return ret;
}

	
template<typename T>
Array<T> varr(const vector<T> &v)
{
    long n = v.size();
    xassert(n > 0);
    
    Array<T> ret({n}, af_uhost);
    memcpy(ret.data, &v[0], n * sizeof(T));
    return ret;
}
    

template<typename T>
struct TestInstance
{
    const int nin;
    const int lmax;
    const int mmax;

    Array<T> theta_big;
    Array<T> phi_big;
    Array<T> wt_big;

    vector<T> theta_small;
    vector<T> phi_small;
    vector<T> wt_small;
    
    TestInstance(int nin_, int lmax_, int mmax_) :
	nin(nin_), lmax(lmax_), mmax(mmax_)
    {
	this->theta_big = rand_array<T> (nin, 0.001 * M_PI, 0.999 * M_PI);
	this->phi_big = rand_array<T> (nin, 0.0, 2*M_PI);
	this->wt_big = Array<T> ({nin}, af_uhost | af_zero);
    }

    void add_point(int ix, T wt, bool noisy=false)
    {
	xassert((ix >= 0) && (ix < nin));

	T theta = theta_big.data[ix];
	T phi = phi_big.data[ix];

	if (noisy)
	    cout << "add_point(): ix=" << ix<< ", theta=" << theta << ", phi=" << phi << ", wt=" << wt << endl;
	
	this->theta_small.push_back(theta);
	this->phi_small.push_back(phi);
	this->wt_small.push_back(wt);
	this->wt_big.data[ix] += wt;
    }

    void add_points(int npoints, bool noisy=false)
    {
	for (int i = 0; i < npoints; i++) {
	    int ix = rand_int(0, nin);
	    T wt = rand_uniform(-1.0, 1.0);
	    this->add_point(ix, wt, noisy);
	}
    }

    double epsabs() const
    {
	double wt2 = 0.0;
	for (ulong i = 0; i < wt_small.size(); i++)
	    wt2 += wt_small[i] * wt_small[i];

	return Eps<T>::eps * (sqrt(wt2 * (lmax+1)) + 0.1);
    }
    
    void run_test(bool announce=true)
    {
	if (announce) {
	    cout << "Running test: dtype=" << type_name<T>()
		 << ", lmax=" << lmax << ", mmax=" << mmax
		 << ", nlg=" << nin << ", nsm=" << theta_small.size()
		 << endl;
	}
	
	int nalm = alm_complex_nelts(lmax, mmax);
	Array<complex<T>> alm_gpu({nalm}, af_gpu | af_random);
	
	launch_points2alm(alm_gpu, theta_big.to_gpu(), phi_big.to_gpu(), wt_big.to_gpu(), lmax, mmax);
	alm_gpu = alm_gpu.to_host();

	// cout << "epsabs=" << epsabs() << endl;
	Array<complex<T>> alm_cpu = reference_points2alm(varr(theta_small), varr(phi_small), varr(wt_small), lmax, mmax);
	assert_arrays_equal(alm_gpu, alm_cpu, "gpu", "cpu", {"ix"}, epsabs(), 0);	  // (epsabs, epsrel)
    }
};
 

void show_ix(int lmax, int mmax, int i)
{
    // FIXME brain-dead
    for (int m = 0; m <= mmax; m++) {
	int a = i - alm_complex_offset(lmax, m);
	
	if ((a < 0) || (a >= (lmax-m+1)))
	    continue;
	
	cout << "i=" << i
	     << ": m=" << m
	     << ", l=" << (m + a)
	     << endl;
	
	return;
    }

    throw runtime_error("out of range");
}

	
int main(int argc, char **argv)
{
    const bool noisy = false;
    
    cout << "Note: in double precision, this cuda program (test-sht.cu) is equivalent to 'python -m direct_sht test'\n"
	 << "However, I'm keeping the C++ program around since it's currently the only way to test the single-precison transforms."
	 << endl;

    for (int i = 0; i < 50; i++) {
	int lmax = rand_int(0, 1001);
	int mmax = rand_int(0, lmax+1);
	int nsm = rand_int(30, 100);
	int nlg = rand_int(nsm, 32768);

	if (i % 2) {
	    TestInstance<double> ti64(nlg, lmax, mmax);
	    ti64.add_points(nsm, noisy);
	    ti64.run_test();
	}
	else {
	    TestInstance<float> ti32(nlg, lmax, mmax);
	    ti32.add_points(nsm, noisy);
	    ti32.run_test();
	}
    }
    
    return 0;
}
