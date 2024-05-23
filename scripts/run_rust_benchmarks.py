#!/usr/bin/env python3
"""Do a multiple-run of the Rust bandwidth benchmark."""
import argparse
import contextlib
from pathlib import Path
import subprocess
import tempfile


def run_benchmark(args, output_file):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as temp_fp:
        print(f'(temporary file {temp_fp.name})')
        print('#!/bin/sh', file=temp_fp)
        print('#SBATCH -N 2', file=temp_fp)
        print(f'#SBATCH -o {output_file}', file=temp_fp)
        print(f'export OMPI_MCA_pml=ucx', file=temp_fp)
        print('mpirun -np 2 -N 1', ' '.join(args), file=temp_fp)
        temp_fp.flush()
        subprocess.run(['sbatch', '-W', temp_fp.name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-path', required=True,
                        help='directory to place results in')
    parser.add_argument('-o', '--options-path', required=True,
                        help='options file path to run the benchmarks with')
    parser.add_argument('-t', '--type', required=True, choices=('bw', 'latency'),
                        help='benchmark type')
    parser.add_argument('-s', '--subvector-size', required=True,
                        help='subvector size to use for benchmarks')
    args = parser.parse_args()

    if args.type == 'bw':
        benchmarks = {
            'rsmpi': ['./target/release/rsmpi_bw'],
            'custom': ['./target/release/mpicd_bw', '--kind', 'custom',
                       '--subvector-size', str(args.subvector_size)],
            'packed': ['./target/release/mpicd_bw', '--kind', 'packed',
                       '--subvector-size', str(args.subvector_size)],
        }
    else:
        benchmarks = {
            'rsmpi': ['./target/release/rsmpi_latency'],
            'custom': ['./target/release/mpicd_latency', '--kind', 'custom',
                       '--subvector-size', str(args.subvector_size)],
            'packed': ['./target/release/mpicd_latency', '--kind', 'packed',
                       '--subvector-size', str(args.subvector_size)],
        }

    for benchmark_name, benchmark_args in benchmarks.items():
        full_args = benchmark_args[:]
        full_args.extend(['--options-path', args.options_path])
        for i in range(4):
            print(f'Running benchmark {benchmark_name}-{i}')
            output_file = str(Path(args.results_path, f'{benchmark_name}-{i}.out'))
            run_benchmark(full_args, output_file)
