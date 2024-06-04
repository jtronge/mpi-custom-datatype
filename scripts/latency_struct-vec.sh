#!/bin/sh

export UCX_RNDV_THRESH=2M
export RESULTS=results/latency_struct-vec
mkdir -p $RESULTS
./scripts/run_rust_benchmarks.py -r $RESULTS -n custom \
    ./target/release/mpicd_latency --kind custom --datatype struct-vec --options-path ./benchmark-options/latency-struct-vec.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n packed \
    ./target/release/mpicd_latency --kind packed --datatype struct-vec --options-path ./benchmark-options/latency-struct-vec.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n rsmpi \
    ./target/release/rsmpi_latency --datatype struct-vec --options-path ./benchmark-options/latency-struct-vec.yml 
