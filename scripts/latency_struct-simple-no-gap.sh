#!/bin/sh

export RESULTS=results/latency_struct-simple-no-gap
mkdir -p $RESULTS
./scripts/run_rust_benchmarks.py -r $RESULTS -n custom \
    ./target/release/mpicd_latency --kind custom --datatype struct-simple-no-gap --options-path ./benchmark-options/latency-struct-simple-no-gap.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n packed \
    ./target/release/mpicd_latency --kind packed --datatype struct-simple-no-gap --options-path ./benchmark-options/latency-struct-simple-no-gap.yml --subvector-size 0
./scripts/run_rust_benchmarks.py -r $RESULTS -n rsmpi-struct-simple-no-gap \
    ./target/release/rsmpi_latency --datatype struct-simple-no-gap --options-path ./benchmark-options/latency-struct-simple-no-gap.yml 
