# Copyright (c) 2012 The Trustees of University of Illinois. All 
# rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

#!/bin/sh

. ./test.inc

NPROCS=$1
OUTER=$2
INNER=$3

OLD_PATH=$PATH
OLD_LD_LIBRARY=$LD_LIBRARY_PATH

LOCAL_PATH=`pwd`

export PATH=${LOCAL_PATH}/local/openmpi-$OMPI_UNSTABLE/bin:$PATH
export LD_LIBRARY_PATH=${LOCAL_PATH}/local/openmpi-$OMPI_UNSTABLE/lib:$LD_LIBRARY_PATH

echo $PATH
echo $LD_LIBRARY_PATH

make clean
make ddtbench_c
mv -f ddtbench.out ddtbench.out.old

mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c.openmpi-$OMPI_UNSTABLE

make ddtbench_f90
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90.openmpi-$OMPI_UNSTABLE

make ddtbench_c_onesided
mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c_onesided.openmpi-$OMPI_UNSTABLE

make ddtbench_f90_onesided
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90_onesided.openmpi-$OMPI_UNSTABLE

export PATH=$OLD_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY

export PATH=${LOCAL_PATH}/local/openmpi-$OMPI_STABLE/bin:$PATH
export LD_LIBRARY_PATH=${LOCAL_PATH}/local/openmpi-$OMPI_STABLE/lib:$LD_LIBRARY_PATH

make clean
make ddtbench_c

mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c.openmpi-$OMPI_STABLE

make ddtbench_f90
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90.openmpi-$OMPI_STABLE

make ddtbench_c_onesided
mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c_onesided.openmpi-$OMPI_STABLE

make ddtbench_f90_onesided
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90_onesided.openmpi-$OMPI_STABLE

export PATH=$OLD_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY

export PATH=${LOCAL_PATH}/local/mpich-$MPICH_UNSTABLE/bin:$PATH
export LD_LIBRARY_PATH=${LOCAL_PATH}/local/mpich-$MPICH_UNSTABLE/lib:$LD_LIBRARY_PATH

make clean
make ddtbench_c

mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c.mpich-$MPICH_UNSTABLE

make ddtbench_f90
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90.mpich-$MPICH_UNSTABLE

make ddtbench_c_onesided
mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c_onesided.openmpi-$MPICH_UNSTABLE

make ddtbench_f90_onesided
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90_onesided.openmpi-$MPICH_UNSTABLE

export PATH=$OLD_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY

export PATH=${LOCAL_PATH}/local/mpich-$MPICH_STABLE/bin:$PATH
export LD_LIBRARY_PATH=${LOCAL_PATH}/local/mpich-$MPICH_STABLE/lib:$LD_LIBRARY_PATH

make clean
make ddtbench_c

mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c.mpich-$MPICH_STABLE

make ddtbench_f90 $OUTER $INNER
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90.mpich-$MPICH_STABLE

make ddtbench_c_onesided
mpiexec -n $NPROCS ./ddtbench_c $OUTER $INNER

mv ddtbench.out ddtbench.out.c_onesided.openmpi-$MPICH_STABLE

make ddtbench_f90_onesided
mpiexec -n $NPROCS ./ddtbench_f90 $OUTER $INNER

mv ddtbench.out ddtbench.out.f90_onesided.openmpi-$MPICH_STABLE

export PATH=$OLD_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY
