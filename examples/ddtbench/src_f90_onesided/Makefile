# Copyright (c) 2012 The Trustees of University of Illinois. All 
# rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

include ../Makefile.inc

DDTBENCH_OBJS = \
	ddtbench.o \
	../src_f90/module_utilities.o \
	../src_f90/module_datatypes.o \
	module_wrapper.o \
	module_timing_nas.o \
	module_timing_wrf.o \
	module_timing_milc.o \
	module_timing_specfem3d.o \
	module_timing_lammps.o \
	module_timing_basic.o \
	module_timing_fft2d.o \
	../src_c/timing.o
#	module_timing.o \

DDTBENCH_MODULES = \
	timing_nas.mod \
	timing_wrf.mod \
	timing_milc.mod \
	timing_specfem3d.mod \
	timing_lammps.mod \
	timing_basic.mod \
	timing_fft2d.mod \
	../src_f90/utilities.mod \
	../src_f90/datatypes.mod \
	wrapper.mod 
#	timing.mod \

# clear out all suffixes
.SUFFIXES:
# list only those we use
.SUFFIXES: .o .F90 .mod

# some implicit rules
.F90.o:
	$(FC) -I../src_f90 $(FCFLAGS) -c $<

%.mod: module_%.F90
	$(FC) -I../src_f90 $(FCFLAGS) -c $<

# some general rules
all: ddtbench

clean:
	rm -f *.mod *.o ddtbench *.lst ../src_c/timing.o ../src_f90/datatypes.mod  ../src_f90/module_datatypes.o ../src_f90/utilities.mod ../src_f90/module_utilities.o

distclean: clean

#dependencies
ddtbench: $(DDTBENCH_OBJS)
	${FC} ${FCFLAGS} -o $@ $(DDTBENCH_OBJS) ${LDFLAGS} ${LIBS}

ddtbench.o: ddtbench.F90 wrapper.mod

timing_nas.mod module_timing_nas.o: module_timing_nas.F90 ../src_f90/utilities.mod

timing_wrf.mod module_timing_wrf.o: module_timing_wrf.F90 ../src_f90/utilities.mod ../src_f90/datatypes.mod

timing_milc.mod module_timing_milc.o: module_timing_milc.F90 ../src_f90/utilities.mod

timing_specfem3d.mod module_timing_specfem3d.o: module_timing_specfem3d.F90 ../src_f90/utilities.mod

timing_lammps.mod module_timing_lammps.o: module_timing_lammps.F90 ../src_f90/utilities.mod

timing_basic.mod module_timing_basic.o: module_timing_basic.F90 ../src_f90/utilities.mod

timing_fft2d.mod module_timing_fft2d.o: module_timing_fft2d.F90 ../src_f90/utilities.mod

../src_c/timing.o: ../src_c/timing.c
	make -C ../src_c timing.o

wrapper.mod module_wrapper.o: ../src_f90/module_wrapper.F90 ../src_f90/utilities.mod timing_basic.mod timing_nas.mod timing_wrf.mod timing_milc.mod timing_specfem3d.mod timing_lammps.mod timing_fft2d.mod
	$(FC) -I../src_f90/ $(FCFLAGS) -c ../src_f90/module_wrapper.F90

../src_f90/datatypes.mod ../src_f90/module_datatypes.o: ../src_f90/module_datatypes.F90
	make -C ../src_f90 module_datatypes.o

../src_f90/utilities.mod ../src_f90/module_utilities.o: ../src_f90/module_utilities.F90
	make -C ../src_f90 module_utilities.o
