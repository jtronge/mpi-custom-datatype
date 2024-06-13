#!/bin/sh
#
# This script runs perf for each rank of an MPI job. Profile data is dumped to
# $PROFILE_DIR; a tool, such as flamegraph, should be used after the job is
# complete to visualize the results.
#
# Example command to run:
#
# `PROFILE_DIR=./struct-vec-profile-packed mpirun -np 2 -N 1 ./scripts/profile.sh ./target/release/mpicd_bw --kind packed --datatype struct-simple --options-path ./benchmark-options/bw-struct-simple-profile.yml --subvector-size 0`
#
# Then to generate a flamegraph:
# flamegraph --perfdata $PROFILE_DIR/perf-0.out -- ./target/release/mpicd_bw
#

if [ -z "$PROFILE_DIR" ]; then
    printf "PROFILE_DIR must be set in environment\n"
    exit 1
fi

if [ -z "$OMPI_COMM_WORLD_RANK" ]; then
    printf "This script must be run with Open MPI's mpirun\n"
    exit 1
fi

exec perf record -F 997 --call-graph dwarf,16384 -g -o $PROFILE_DIR/perf-$OMPI_COMM_WORLD_RANK.out -- $@
