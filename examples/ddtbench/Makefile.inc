# Copyright (c) 2012 The Trustees of University of Illinois. All
# rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

FC=mpif90
CC=mpicc
CXX=mpicxx

PAPI_INSTALL_PATH=

HRT_ARCH=6   # 1 : x86_32, gcc, rdtsc
             # 2 : x64_64, gcc, rdtsc
             # 3 : ppc, gcc, tb
             # 4 : ia64, gcc, itc
             # 5 : mips64-sicortex, gcc
             # 6 : mpi_wtime

TEST_TYPE=1  # 1: time
             # 2: papi
             # 3: time+papi

OPTIMIZATION=-O3
DEBUG=
WARNINGS=-Wall

FCFLAGS+=${OPTIMIZATION} ${DEBUG} ${WARNINGS} -DTEST_TYPE=${TEST_TYPE} -I../src_f90/
CCFLAGS+=${OPTIMIZATION} ${DEBUG} ${WARNINGS} -DHRT_ARCH=${HRT_ARCH} -DTEST_TYPE=${TEST_TYPE}

LDFLAGS=
LIBS= #-lpapi

ifneq ($(PAPI_INSTALL_PATH),)
	CCFLAGS+=-I$(PAPI_INSTALL_PATH)/include
	LDFLAGS+=-L$(PAPI_INSTALL_PATH)/lib
endif
