#!/bin/sh
#SBATCH -o results-2023-03-22/complex-compound_iovec_3_server.out
#SBATCH -N 1
#SBATCH -w er03
source /home/jtronge/ompi-install2/env
target/release/latency_iovec 172.16.0.3 -p 1347 -c ./inputs/complex-compound.yaml -s
