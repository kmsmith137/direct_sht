// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in gputils/src_pybind11/gputils_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_direct_sht

#include <iostream>
#include <gputils/pybind11.hpp>
#include "../include/direct_sht.hpp"


using namespace std;
using namespace gputils;
namespace py = pybind11;


static void _launch_direct_sht(Array<complex<double>> &out_alm, const Array<double> &in_theta, const Array<double> &in_phi, const Array<double> &wt, int lmax, int mmax, int stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t> (stream_ptr);
    direct_sht::launch_direct_sht(out_alm, in_theta, in_phi, wt, lmax, mmax, stream);
}

PYBIND11_MODULE(direct_sht_pybind11, m)  // extension module gets compiled to direct_sht_pybind11.so
{
    m.doc() = "direct_sht: \"direct\" spherical transforms on GPUs";

    // Note: looks like _import_array() will fail if different numpy versions are
    // found at compile-time versus runtime.

    if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "direct_sht: numpy.core.multiarray failed to import");
	return;
    }
    
    m.def("_launch_direct_sht", &_launch_direct_sht,
	  // "docstring",
	  py::arg("out_alm"), py::arg("in_theta"), py::arg("in_phi"), py::arg("wt"), py::arg("lmax"), py::arg("mmax"), py::arg("stream_ptr"));
}
