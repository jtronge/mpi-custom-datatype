# Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

include ../Makefile.inc

DDTBENCH_OBJS = \
	ddtbench.o \
	../src_c/utilities.o \
	../src_c/timing.o \
	../src_c/wrapper.o \
	timing_basic.o \
	timing_fft2d.o \
	timing_lammps.o \
	timing_milc.o \
	timing_nas.o \
	timing_specfem3d.o \
	timing_wrf.o

# clear out all suffixes
.SUFFIXES:
# list only those we use
.SUFFIXES: .o .c

# some implicit rules
.c.o:
	$(CC) $(CCFLAGS) -c $<

# some general rules
all: ddtbench

distclean: clean

clean:
	rm -f *.o ddtbench ../src_c/utilities.o ../src_c/timing.o ../src_c/wrapper.o

ddtbench: $(DDTBENCH_OBJS)
	${CC} ${CCFLAGS} -o $@ $(DDTBENCH_OBJS) ${LDFLAGS} ${LIBS}

../src_c/utilities.o: ../src_c/utilities.c
	make -C ../src_c utilities.o

../src_c/timing.o: ../src_c/timing.c
	make -C ../src_c timing.o

../src_c/wrapper.o: ../src_c/wrapper.c
	make -C ../src_c wrapper.o
