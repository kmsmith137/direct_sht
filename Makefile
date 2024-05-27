GPUTILS_INCDIR=../gputils/include
GPUTILS_LIBDIR=../gputils/lib

ARCH =
ARCH += -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 -I$(GPUTILS_INCDIR) --compiler-options -Wall,-fPIC
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/direct_sht.hpp

OFILES = \
  src_lib/gpu_kernel.o \
  src_lib/reference_sht.o

LIBFILES = \
  lib/libdirect_sht.a \
  lib/libdirect_sht.so

XFILES = \
  bin/test-sht \
  bin/time-sht

SRCDIRS = \
  include \
  include/direct_sht \
  src_bin \
  src_lib

all: $(LIBFILES) $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) $(LIBFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

bin/%: src_bin/%.o lib/libdirect_sht.a $(GPUTILS_LIBDIR)/libgputils.a
	mkdir -p bin && $(NVCC) -o $@ $^

lib/libdirect_sht.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

lib/libdirect_sht.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^

#INSTALL_DIR ?= /usr/local
#
#install: $(LIBFILES)
#	mkdir -p $(INSTALL_DIR)/include
#	mkdir -p $(INSTALL_DIR)/lib
#	cp -rv lib $(INSTALL_DIR)/
#	cp -rv include $(INSTALL_DIR)/
