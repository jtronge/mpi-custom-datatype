#!/bin/sh
#SBATCH -o results-2023-03-22/complex-compound_iovec_1_client.out
#SBATCH -N 1
source /home/jtronge/ompi-install3/env
target/release/latency_iovec 172.16.0.4 -p 1347 -c ./inputs/complex-compound.yaml
