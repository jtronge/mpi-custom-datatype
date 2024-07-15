#!/bin/sh

export UCX_NET_DEVICES="mlx5_0:1"
export ADDITIONAL_ARGS="-m 1048576 -n 1073741824"

mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ADDITIONAL_ARGS \
    > results/python-pingpong-complex-object/two-node/baseline.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ADDITIONAL_ARGS -poc --array complex-object \
    > results/python-pingpong-complex-object/two-node/pickle_oob_cdt.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ADDITIONAL_ARGS -po --array complex-object \
    > results/python-pingpong-complex-object/two-node/pickle_oob.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ADDITIONAL_ARGS -p --array complex-object \
    > results/python-pingpong-complex-object/two-node/pickle_basic.out
