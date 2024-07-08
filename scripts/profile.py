#!/usr/bin/env python3
"""Profile an MPI job on all ranks."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


def run_profile(args, use_slurm=False):
    """Run a profile for the program, yielding a list of profile files."""
    profile_path = Path(os.path.expanduser('~/.profiles'))
    temp_dir = profile_path / uuid.uuid4().hex
    os.makedirs(temp_dir)
    print(f'profile dir: {temp_dir}')
    sbatch_script = Path(temp_dir, 'sbatch.sh')
    cmd_script = Path(temp_dir, 'command.sh')
    # Generate the command script with perf wrapping the real program to be run
    perf_cmd = f'perf record -F 997 --call-graph dwarf,16384 -g -o {temp_dir}/perf.$OMPI_COMM_WORLD_RANK'
    with open(cmd_script, 'w') as cmd_fp:
        cmd = ' '.join(args)
        print('#!/bin/sh', file=cmd_fp)
        print(f'exec {perf_cmd} -- {cmd}', file=cmd_fp)

    if use_slurm:
        # Create the sbatch script that invokes the command script
        output_log = f'{temp_dir}/out.log'
        with open(sbatch_script, 'w') as sbatch_fp:
            print('#!/bin/sh', file=sbatch_fp)
            print('#SBATCH -N 2', file=sbatch_fp)
            print(f'#SBATCH -o {output_log}', file=sbatch_fp)
            print(f'mpirun -np 2 -N 1 /bin/sh {cmd_script}', file=sbatch_fp)

        # Now wait for the job to complete
        subprocess.run(['sbatch', '-W', sbatch_script], check=True)
    else:
        subprocess.run(['mpirun', '-np', '2', '/bin/sh', cmd_script], check=True)

    # Find all the perf outputs
    for fname in os.listdir(temp_dir):
        if not fname.startswith('perf.'):
            continue
        _, rank = fname.split('.')
        yield temp_dir / fname, int(rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output directory for flamegraphs')
    parser.add_argument('-s', '--slurm', action='store_true', help='use a slurm batch script (default to two nodes)')
    args, prog_args = parser.parse_known_args()

    for profile, rank in run_profile(prog_args, use_slurm=args.slurm):
        subprocess.run(['flamegraph', '--perfdata', profile, '-o',
                        Path(args.output, f'flamegraph-{rank}.svg'), '--',
                        prog_args[0]], check=True)
