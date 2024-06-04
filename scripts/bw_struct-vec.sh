#!/bin/sh

export UCX_RNDV_THRESH=2M
export RESULTS=results/bw_struct-vec
mkdir -p $RESULTS
./scripts/run_rust_benchmarks.py -r $RESULTS -n custom \
    ./target/release/mpicd_bw --kind custom --datatype struct-vec --options-path ./benchmark-options/bw-struct-vec.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n packed \
    ./target/release/mpicd_bw --kind packed --datatype struct-vec --options-path ./benchmark-options/bw-struct-vec-packed.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n rsmpi-struct-vec \
    ./target/release/rsmpi_bw --datatype struct-vec --options-path ./benchmark-options/bw-struct-vec.yml 
