project(
    'direct_sht',
    'cuda', 'cpp',
     default_options: ['python.install_env=auto','optimization=3','debug=false','b_ndebug=false','cpp_std=c++17']
)

header_files_toplevel = [
    'include/direct_sht.hpp'
]

lib_source_files = [
    'src_lib/gpu_kernel.cu',
    'src_lib/reference_sht.cu'
]

pyext_source_files = [
    'src_pybind11/direct_sht_pybind11.cu'
]

python_files_toplevel = [
    'src_python/direct_sht/__init__.py'
]    

local_executables = [
    'test-sht',
    'time-sht'
]


####################################################################################################


fs = import('fs')
cpp = meson.get_compiler('cpp')

# Not sure if this is really necessary
cuda_args = ['-m64']
link_args = ['-m64']

py3 = import('python').find_installation('python3', pure: false)
py3_install_dir = py3.get_install_dir()
# message('py3.path = ', py3.path())
# message('py3.install_dir = ', py3_install_dir)

# Show sys.path for debugging
# ret = run_command(py3.path(), '-c', 'import sys; print(sys.path)', check: true)
# message('Debug: py3.sys.path = ', ret.stdout().strip())


####################################################################################################


gputils_ret = run_command(py3.path(), '-c', 'import gputils; print(gputils.__file__)', check: true)
gputils_ret = gputils_ret.stdout().strip()
message('gputils_ret = ', gputils_ret)

gputils_incdir = fs.parent(gputils_ret) / 'include'
gputils_libdir = fs.parent(gputils_ret)

if not fs.exists(gputils_incdir / 'gputils.hpp')
    error('Couldn\'t find gputils.hpp in expected location -- time to debug')
endif

if not fs.exists(gputils_libdir / 'libgputils.so')
    error('Couldn\'t find libgputils.so in expected location -- time to debug')
endif

gputils_pyext_ret = run_command(py3.path(), '-c', 'import gputils; print(gputils.gputils_pybind11.__file__)', check: true)
gputils_pyext_ret = gputils_pyext_ret.stdout().strip()
message('gputils_pyext_ret = ', gputils_pyext_ret)

gputils_pyext_dir = fs.parent(gputils_pyext_ret)
gputils_pyext_basename = fs.name(gputils_pyext_ret)
# message('gputils_pyext_dir = ', gputils_pyext_dir)
# message('gputils_pyext_basename =', gputils_pyext_basename)

if not gputils_pyext_basename.endswith('.so')
    error('gputils.gputils_pybind11.__file__ did not end with .so as expected -- time to debug')
endif

gputils_pyext_stem = gputils_pyext_basename.substring(0,-3)
# message('gputils_pyext_stem =', gputils_pyext_stem)

# FIXME using cpp compiler here, since using cuda compiler doesn't seem to work!
gputils_lib = cpp.find_library('gputils', dirs: gputils_libdir)
gputils_pyext = cpp.find_library(gputils_pyext_stem, dirs: gputils_libdir)


####################################################################################################


pybind11_ret = run_command(py3.path(), '-m', 'pybind11', '--includes', check: true)
pybind11_ret = pybind11_ret.stdout().strip()
# message('Debug: Output of python -m pybind11 --includes: ', pybind11_ret)

pybind11_include_dirs = [ ]
foreach include_dir: pybind11_ret.split()
    include_dir = include_dir.strip().strip('-I')
    pybind11_include_dirs += include_dir
endforeach

# message('Debug: pybind11_include_dirs = ', pybind11_include_dirs)
pybind11_dep = declare_dependency(include_directories: pybind11_include_dirs)


####################################################################################################


numpy_incdir = run_command(py3.path(), '-c', 'import numpy; print(numpy.get_include())', check: true)
numpy_incdir = numpy_incdir.stdout().strip()

# message('Debug: numpy.get_include() = ', numpy_incdir)
numpy_dep = declare_dependency(include_directories: numpy_incdir)


####################################################################################################


direct_sht_lib = library(
    'direct_sht',
    lib_source_files,
    cuda_args: cuda_args,
    link_args: link_args,
    include_directories: gputils_incdir,
    dependencies: [gputils_lib],
    install: true,
    install_dir: py3.get_install_dir() / 'direct_sht'
)

py3.extension_module(
    'direct_sht_pybind11',   # compiled extension will be named direct_sht_pybind11.so
    sources: pyext_source_files,
    dependencies : [pybind11_dep, numpy_dep, gputils_pyext, gputils_lib],
    include_directories: gputils_incdir,
    cuda_args: cuda_args,
    link_args: link_args,
    link_language : 'cuda',
    link_with: direct_sht_lib,  # link libdirect_sht.so to direct_sht_pybind11.so
    override_options: ['cpp_rtti=true'],
    subdir: 'direct_sht',
    install: true
)

py3.install_sources(
    python_files_toplevel,
    subdir: 'direct_sht'
)

py3.install_sources(
    header_files_toplevel,
    subdir: 'direct_sht/include'
)

foreach x: local_executables
    executable(
        x,
        'src_bin/' + x + '.cu',
	dependencies: [gputils_lib],
	include_directories: gputils_incdir,
        cuda_args: cuda_args,
	link_args: link_args,
	link_with: direct_sht_lib
    )
endforeach
