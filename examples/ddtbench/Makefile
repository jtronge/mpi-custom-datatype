# Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

include Makefile.inc
#some parameter for the test environment, in a different file, so that the shells scripts can red them properly
include test.inc
#all: ddtbench_c ddtbench_f90 ddtbench_c_onesided ddtbench_f90_onesided
#all: ddtbench_c_custom ddtbench_c ddtbench_f90 ddtbench_c_onesided ddtbench_f90_onesided
all: ddtbench_c_custom ddtbench_c
#all: ddtbench_c_custom

clean:
	$(MAKE) -C src_f90 clean
	$(MAKE) -C src_c_custom clean
	$(MAKE) -C src_c clean
	$(MAKE) -C src_c_onesided clean
	$(MAKE) -C src_f90_onesided clean
	rm -f ddtbench_c ddtbench_c_custom ddtbench_f90 ddtbench_c_onesided ddtbench_f90_onesided

distclean: clean
	$(MAKE) -C src_f90 distclean
	$(MAKE) -C src_c distclean
	$(MAKE) -C src_c_custom distclean
	$(MAKE) -C src_c_onesided distclean
	$(MAKE) -C src_f90_onesided distclean
	rm -rf src_mpi
	rm -rf local

ddtbench_c:
	$(MAKE) -C src_c ddtbench
	mv src_c/ddtbench ddtbench_c

ddtbench_c_custom:
	$(MAKE) -C src_c_custom ddtbench
	mv src_c_custom/ddtbench ddtbench_c_custom


ddtbench_f90:
	$(MAKE) -C src_f90 ddtbench
	mv src_f90/ddtbench ddtbench_f90

ddtbench_c_onesided:
	$(MAKE) -C src_c_onesided ddtbench
	mv src_c_onesided/ddtbench ddtbench_c_onesided

ddtbench_f90_onesided:
	$(MAKE) -C src_f90_onesided ddtbench
	mv src_f90_onesided/ddtbench ddtbench_f90_onesided

######################################################################################################
# part of the makefile for the fullrun test

test: helper_scripts/fullrun.sh build_mpis
	helper_scripts/fullrun.sh $(NPROCS) $(OUTER) $(INNER)

# get major version (because the download url requires it)
OMPI_STABLE_MAJOR=$(shell echo $(OMPI_STABLE) | sed -e 's/\([0-9].[0-9]\).*/\1/';)
OMPI_UNSTABLE_MAJOR=$(shell echo $(OMPI_UNSTABLE) | sed -e 's/\([0-9].[0-9]\).*/\1/';)

build_mpis: openmpi_stable openmpi_unstable mpich_stable mpich_unstable

mpiclean:
	rm -rf local/*
	$(MAKE) -C src_mpi/openmpi-$(OMPI_STABLE)/ clean
	$(MAKE) -C src_mpi/openmpi-$(OMPI_UNSTABLE)/ clean
	$(MAKE) -C src_mpi/mpich-$(MPICH_STABLE)/ clean
	$(MAKE) -C src_mpi/mpich-$(MPICH_UNSTABLE)/ clean

openmpi_unstable: local/openmpi-$(OMPI_UNSTABLE)/bin/mpif90 local/openmpi-$(OMPI_UNSTABLE)/bin/mpiexec

local/openmpi-$(OMPI_UNSTABLE)/bin/mpif90 local/openmpi-$(OMPI_UNSTABLE)/bin/mpiexec: src_mpi/openmpi-$(OMPI_UNSTABLE).tar.bz2
	helper_scripts/install_ompi.sh openmpi-$(OMPI_UNSTABLE)

src_mpi/openmpi-$(OMPI_UNSTABLE).tar.bz2:
	@echo "Trying to download Open MPI version $(OMPI_UNSTABLE) from http://www.open-mpi.org/software/ompi/v$(OMPI_UNSTABLE_MAJOR)/downloads/openmpi-$(OMPI_UNSTABLE).tar.bz2"
	@mkdir -p src_mpi
	@cd src_mpi ; \
	wget -q http://www.open-mpi.org/software/ompi/v$(OMPI_UNSTABLE_MAJOR)/downloads/openmpi-$(OMPI_UNSTABLE).tar.bz2 ; \
	cd ..

openmpi_stable: local/openmpi-$(OMPI_STABLE)/bin/mpif90 local/openmpi-$(OMPI_STABLE)/bin/mpiexec

local/openmpi-$(OMPI_STABLE)/bin/mpif90 local/openmpi-$(OMPI_STABLE)/bin/mpiexec: src_mpi/openmpi-$(OMPI_STABLE).tar.bz2
	helper_scripts/install_ompi.sh openmpi-$(OMPI_STABLE)

src_mpi/openmpi-$(OMPI_STABLE).tar.bz2:
	@echo "Trying to download Open MPI version $(OMPI_STABLE) from http://www.open-mpi.org/software/ompi/v$(OMPI_STABLE_MAJOR)/downloads/openmpi-$(OMPI_STABLE).tar.bz2"
	@mkdir -p src_mpi
	@cd src_mpi ; \
	wget -q http://www.open-mpi.org/software/ompi/v$(OMPI_STABLE_MAJOR)/downloads/openmpi-$(OMPI_STABLE).tar.bz2 ; \
	cd ..

mpich_stable: local/mpich-$(MPICH_STABLE)/bin/mpif90 local/mpich-$(MPICH_STABLE)/bin/mpiexec

local/mpich-$(MPICH_STABLE)/bin/mpif90 local/mpich-$(MPICH_STABLE)/bin/mpiexec: src_mpi/mpich-$(MPICH_STABLE).tar.gz
	helper_scripts/install_mpich.sh mpich-$(MPICH_STABLE)

src_mpi/mpich-$(MPICH_STABLE).tar.gz:
	@echo "Trying to download MPICH version $(MPICH_STABLE) from http://www.mpich.org/static/downloads/$(MPICH_STABLE)/mpich-$(MPICH_STABLE).tar.gz"
	@mkdir -p src_mpi
	@cd src_mpi ; \
	wget -q http://www.mpich.org/static/downloads/$(MPICH_STABLE)/mpich-$(MPICH_STABLE).tar.gz ; \
	cd ..

mpich_unstable: local/mpich-$(MPICH_UNSTABLE)/bin/mpif90 local/mpich-$(MPICH_UNSTABLE)/bin/mpiexec

local/mpich-$(MPICH_UNSTABLE)/bin/mpif90 local/mpich-$(MPICH_UNSTABLE)/bin/mpiexec: src_mpi/mpich-$(MPICH_UNSTABLE).tar.gz
	helper_scripts/install_mpich.sh mpich-$(MPICH_UNSTABLE)

src_mpi/mpich-$(MPICH_UNSTABLE).tar.gz:
	@echo "Trying to download MPICH version $(MPICH_UNSTABLE) from http://www.mpich.org/static/downloads/$(MPICH_UNSTABLE)/mpich-$(MPICH_UNSTABLE).tar.gz"
	@mkdir -p src_mpi
	@cd src_mpi ; \
	wget http://www.mpich.org/static/downloads/$(MPICH_UNSTABLE)/mpich-$(MPICH_UNSTABLE).tar.gz ; \
	cd ..
