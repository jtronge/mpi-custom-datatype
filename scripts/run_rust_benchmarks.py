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
        # print('prterun -np 2 -N 1', ' '.join(args), file=temp_fp)
        print('mpirun -np 2 -N 1', ' '.join(args), file=temp_fp)
        temp_fp.flush()
        subprocess.run(['sbatch', '-W', temp_fp.name], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-path', required=True,
                        help='directory to place results in')
    parser.add_argument('-n', '--name', required=True,
                        help='name of benchmark to run')
    args, prog_args = parser.parse_known_args()

    for i in range(4):
        print(f'Running benchmark {args.name}-{i}')
        output_file = str(Path(args.results_path, f'{args.name}-{i}.out'))
        run_benchmark(prog_args, output_file)
