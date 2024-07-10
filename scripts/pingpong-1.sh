#!/bin/sh

export UCX_TLS=rc
mpirun -np 2 python3 ./examples/pingpong.py > results/python-pingpong/one-node/baseline.out
mpirun -np 2 python3 ./examples/pingpong.py -poc > results/python-pingpong/one-node/pickle_oob_cdt.out
mpirun -np 2 python3 ./examples/pingpong.py -po > results/python-pingpong/one-node/pickle_oob.out
mpirun -np 2 python3 ./examples/pingpong.py -p > results/python-pingpong/one-node/pickle_basic.out
