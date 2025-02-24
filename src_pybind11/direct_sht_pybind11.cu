// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_direct_sht

#include <iostream>
#include <ksgpu/pybind11.hpp>
#include "../include/direct_sht.hpp"


using namespace std;
using namespace ksgpu;
namespace py = pybind11;


static void _launch_points2alm(Array<complex<double>> &out_alm, const Array<double> &in_theta, const Array<double> &in_phi, const Array<double> &wt, int lmax, int mmax, int stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t> (stream_ptr);
    direct_sht::launch_points2alm(out_alm, in_theta, in_phi, wt, lmax, mmax, stream);
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
    
    m.def("launch_points2alm", &_launch_points2alm,
	  "All arrays must be on the GPU.",
	  py::arg("out_alm"), py::arg("theta"), py::arg("phi"), py::arg("wt"), py::arg("lmax"), py::arg("mmax"), py::arg("stream_ptr"));

    m.def("reference_points2alm", &direct_sht::reference_points2alm<double>,
	  "Slow, reference implementation of points2alm. All arrays must be on the CPU. Returns a complex alm array.",
	  py::arg("theta"), py::arg("phi"), py::arg("wt"), py::arg("lmax"), py::arg("mmax"));
}
