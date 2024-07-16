#!/bin/sh

export UCX_NET_DEVICES="mlx5_0:1"
export ARGS="-m 2097152 -n 1073741824 --complex-object-size 2097152"
export RESULTS_PATH=results/python-pingpong-complex-object-2mib-3/two-node

mkdir -p $RESULTS_PATH
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ARGS \
    > $RESULTS_PATH/baseline.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ARGS -poc --array complex-object \
    > $RESULTS_PATH/pickle_oob_cdt.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ARGS -po --array complex-object \
    > $RESULTS_PATH/pickle_oob.out
mpirun -np 2 -N 1 -- python3 ./examples/pingpong.py $ARGS -p --array complex-object \
    > $RESULTS_PATH/pickle_basic.out
