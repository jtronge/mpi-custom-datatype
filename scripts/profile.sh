#!/bin/sh

# Install flamegraph with `cargo install flamegraph`
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
    exec flamegraph -o out.svg -- $@
else
    exec $@
fi
