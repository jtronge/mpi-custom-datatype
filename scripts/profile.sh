#!/bin/sh

if [ -z "$FLAMEGRAPH_PATH" -o -z "$FLAMEGRAPH_RANK" ]; then
    printf "Both FLAMEGRAPH_PATH and $FLAMEGRAPH_RANK must be set in environment\n"
    exit 1
fi

# Install flamegraph with `cargo install flamegraph`
if [ $OMPI_COMM_WORLD_RANK -eq $FLAMEGRAPH_RANK ]; then
    exec flamegraph -o $FLAMEGRAPH_PATH-$OMPI_COMM_WORLD_RANK.svg -- $@
else
    exec $@
fi
