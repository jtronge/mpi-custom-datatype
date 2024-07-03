# Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


CC=gcc
CXX=g++ -std=c++20 -g

MPICD_ROOT=$(HOME)/opt/mpicd/

HRT_ARCH=1

HRT_ARCH=6   # 1 : x86_32, gcc, rdtsc
             # 2 : x64_64, gcc, rdtsc
             # 3 : ppc, gcc, tb
             # 4 : ia64, gcc, itc
             # 5 : mips64-sicortex, gcc
             # 6 : mpi_wtime

TEST_TYPE=1  # 1: time
             # 2: papi
             # 3: time+papi

OPTIMIZATION=-O3 -march=native -ftree-vectorize
DEBUG=
WARNINGS=-Wall

FCFLAGS+=${OPTIMIZATION} ${DEBUG} ${WARNINGS} -DTEST_TYPE=${TEST_TYPE} -I../src_f90/
CCFLAGS+=${OPTIMIZATION} ${DEBUG} ${WARNINGS} -DHRT_ARCH=${HRT_ARCH} -DTEST_TYPE=${TEST_TYPE}
CCFLAGS+=-I$(MPICD_ROOT)/include

LDFLAGS+=-L$(MPICD_ROOT)/lib -lmpi

DDTBENCH_OBJS = \
	ddtbench.o \
	utilities.o \
	wrapper.o \
	timing.o \
	timing_basic.o \
	timing_lammps_custom.o \
	timing_wrf_custom.o \
	timing_milc_custom.o \
	timing_nas_custom.o

# clear out all suffixes
.SUFFIXES:
# list only those we use
.SUFFIXES: .o .c .cc

# some implicit rules
.c.o:
	$(CC) $(CCFLAGS) -c $<

# some implicit rules
.cc.o:
	$(CXX) $(CCFLAGS) -c $<

# some general rules
all: ddtbench

distclean: clean

clean:
	rm -f *.o ddtbench

ddtbench: $(DDTBENCH_OBJS)
	${CXX} ${CCFLAGS} -o $@ $(DDTBENCH_OBJS) ${LDFLAGS} ${LIBS}
