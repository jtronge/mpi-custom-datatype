#!/bin/sh
# Run the varying subvec size latency benchmark for the double-vec type

export RESULTS=results/latency_double-vec
export SUBVEC_SIZES="64 256 1024 4096"
mkdir -p $RESULTS
# Custom type
for size in $SUBVEC_SIZES; do
    mkdir $RESULTS/custom-$size
    ./scripts/run_rust_benchmarks.py -r $RESULTS/custom-$size -n result \
        ./target/release/mpicd_latency --kind custom \
                                       --datatype double-vec \
                                       --options-path ./benchmark-options/latency-1mb.yml \
                                       --subvector-size $size
done
# Packed type
for size in $SUBVEC_SIZES; do
    mkdir $RESULTS/packed-$size
    ./scripts/run_rust_benchmarks.py -r $RESULTS/packed-$size -n result \
        ./target/release/mpicd_latency --kind packed \
                                       --datatype double-vec \
                                       --options-path ./benchmark-options/latency-1mb.yml \
                                       --subvector-size $size
done
# Bytes baseline
mkdir -p $RESULTS/rsmpi-bytes
./scripts/run_rust_benchmarks.py -r $RESULTS/rsmpi-bytes -n result \
    ./target/release/rsmpi_latency --datatype bytes --options-path ./benchmark-options/latency-1mb.yml 
