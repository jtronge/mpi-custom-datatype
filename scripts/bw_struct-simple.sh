#!/bin/sh

export UCX_RNDV_THRESH=2M
export RESULTS=results/bw_struct-simple
mkdir -p $RESULTS
./scripts/run_rust_benchmarks.py -r $RESULTS -n custom \
    ./target/release/mpicd_bw --kind custom --datatype struct-simple --options-path ./benchmark-options/bw-struct-simple.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n packed \
    ./target/release/mpicd_bw --kind packed --datatype struct-simple --options-path ./benchmark-options/bw-struct-simple-packed.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n rsmpi \
    ./target/release/rsmpi_bw --datatype struct-simple --options-path ./benchmark-options/bw-struct-simple-rsmpi.yml 
