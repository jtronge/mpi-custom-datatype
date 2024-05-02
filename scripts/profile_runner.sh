#!/bin/sh

export PROFILE_RANK=0
mpirun -np 2 ./scripts/profile.sh $@
perf script > out.perf
$FLAME_GRAPH_PATH/stackcollapse-perf.pl out.perf > out.stack
$FLAME_GRAPH_PATH/flamegraph.pl out.stack > out.svg
