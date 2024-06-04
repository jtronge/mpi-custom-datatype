# Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

include ../Makefile.inc

DDTBENCH_OBJS = \
	ddtbench.o \
	utilities.o \
	timing.o \
	wrapper.o \
	timing_nas.o \
	timing_basic.o \
	timing_specfem3d.o \
	timing_lammps.o \
	timing_milc.o \
	timing_fft2d.o \
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
	rm -f *.o ddtbench

ddtbench: $(DDTBENCH_OBJS)
	${CC} ${CCFLAGS} -o $@ $(DDTBENCH_OBJS) ${LDFLAGS} ${LIBS}
