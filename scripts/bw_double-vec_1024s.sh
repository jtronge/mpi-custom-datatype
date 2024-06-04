#!/bin/sh

export RESULTS=results/bw_double-vec_1024s
mkdir -p $RESULTS
./scripts/run_rust_benchmarks.py -r $RESULTS -n custom \
    ./target/release/mpicd_bw --kind custom --datatype double-vec --options-path ./benchmark-options/bw-1mb.yml --subvector-size 1024
./scripts/run_rust_benchmarks.py -r $RESULTS -n packed \
    ./target/release/mpicd_bw --kind packed --datatype double-vec --options-path ./benchmark-options/bw-1mb.yml --subvector-size 1024
./scripts/run_rust_benchmarks.py -r $RESULTS -n rsmpi \
    ./target/release/rsmpi_bw --datatype bytes --options-path ./benchmark-options/bw-1mb.yml 
