#!/bin/sh

mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py > results/python-pingpong-complex-object/two-node/baseline.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py -poc --array complex-object > results/python-pingpong-complex-object/two-node/pickle_oob_cdt.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py -po --array complex-object > results/python-pingpong-complex-object/two-node/pickle_oob.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py -p --array complex-object > results/python-pingpong-complex-object/two-node/pickle_basic.out
