#!/usr/bin/env python
#
# This script is invoked by the Makefile, to generate "derived" Makefile variables
# PYTHON_INCDIR, NUMPY_INCDIR, PYBIND11_INCDIR, PYEXT_SUFFIX, KSGPU_DIR.
#
# Values of these variables are written to makefile_helper.out.


import io
import os
import sysconfig
import importlib.util

try:
    import pybind11
except:
    raise RuntimeError("Couldn't import pybind11 -- this is a fatal error")

try:
    import numpy
except:
    raise RuntimeError("Couldn't import numpy -- this is a fatal error")


s = io.StringIO()
print("# Autogenerated by makefile_helper.py (which is automatically invoked by the Makefile)\n", file=s)

python_incdir = sysconfig.get_config_var('INCLUDEPY')
print("# Include directory for python headers", file=s)
print("# From sysconfig.get_config_var('INCLUDEPY')", file=s)
print(f"PYTHON_INCDIR = {python_incdir}\n", file=s)

numpy_incdir = numpy.get_include()
print("# Include directory for numpy headers", file=s)
print("# From numpy.get_include()", file=s)
print(f"NUMPY_INCDIR = {numpy_incdir}\n", file=s)

pybind11_incdir = pybind11.get_include()
print("# Include directory for pybind11 headers", file=s)
print("# From pybind11.get_include()", file=s)
print(f"PYBIND11_INCDIR = {pybind11_incdir}\n", file=s)

pyext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
print("# Filename suffix for python extension modules", file=s)
print("# From sysconfig.get_config_var('EXT_SUFFIX')", file=s)
print("# Equivalent to 'python3-config --extension-suffix'", file=s)
print(f"PYEXT_SUFFIX = {pyext_suffix}\n", file=s)

# This way of getting KSGPU_DIR is awkward, but avoids importing 'ksgpu'.
# (This is useful since imports can generate unexpected output.)

ksgpu_spec = importlib.util.find_spec('ksgpu')
if ksgpu_spec is None:
    raise RuntimeError("Couldn't find 'ksgpu' package -- this is a fatal error")

ksgpu_dir = os.path.dirname(ksgpu_spec.origin)
print("# Base directory for 'ksgpu' package", file=s)
print("# From importlib.util.find_spec('ksgpu').origin", file=s)
print(f"KSGPU_DIR = {ksgpu_dir}", file=s)

s = s.getvalue()
outfile = 'makefile_helper.out'
print(f'Writing {outfile}')

with open(outfile,'w') as f:
    print(s, file=f, end='')

# Show summary on stdout (for debugging).
if False:
    for line in s.split('\n'):
        if (len(line) > 0) and (line[0] != '#'):
            print('   ', line)
