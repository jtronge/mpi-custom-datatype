#!/bin/sh

if [ $OMPI_COMM_WORLD_RANK -eq $PROFILE_RANK ]; then
    exec perf record -F 99 -g -- $@
else
    exec $@
fi
