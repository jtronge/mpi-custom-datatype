#!/bin/sh

if [ -z "$FLAMEGRAPH_PATH" ]; then
    printf "FLAMEGRAPH_PATH must be set in environment\n"
    exit 1
fi

# Install flamegraph with `cargo install flamegraph`
exec flamegraph -o $FLAMEGRAPH_PATH-$OMPI_COMM_WORLD_RANK.svg -- $@
