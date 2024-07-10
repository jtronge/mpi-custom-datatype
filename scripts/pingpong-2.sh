#!/bin/sh

mpirun -np 2 -N 1 python3 ./examples/pingpong.py > results/python-pingpong/two-node/baseline.out
mpirun -np 2 -N 1 python3 ./examples/pingpong.py -poc > results/python-pingpong/two-node/pickle_oob_cdt.out
mpirun -np 2 -N 1 python3 ./examples/pingpong.py -po > results/python-pingpong/two-node/pickle_oob.out
mpirun -np 2 -N 1 python3 ./examples/pingpong.py -p > results/python-pingpong/two-node/pickle_basic.out
