#!/bin/sh

export UCX_TLS=rc
export NUMPY_MADVISE_HUGEPAGE=0
mpirun -np 2 --bind-to core python3 ./examples/pingpong.py > results/python-pingpong-numpy-var/one-node/baseline.out
mpirun -np 2 --bind-to core python3 ./examples/pingpong.py -poc > results/python-pingpong-numpy-var/one-node/pickle_oob_cdt.out
mpirun -np 2 --bind-to core python3 ./examples/pingpong.py -po > results/python-pingpong-numpy-var/one-node/pickle_oob.out
mpirun -np 2 --bind-to core python3 ./examples/pingpong.py -p > results/python-pingpong-numpy-var/one-node/pickle_basic.out
