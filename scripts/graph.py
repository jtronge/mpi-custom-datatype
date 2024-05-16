#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from matplotlib import pyplot as plt
import numpy as np


def load_result(fname):
    """Load a two-column results from an output file."""
    sizes = []
    values = []
    with open(fname) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            size, value = line.split()
            sizes.append(int(size))
            values.append(float(value))
    return np.array(sizes), np.array(values)


def load_average_results(results_path, prefix):
    """Load and average multiple runs in the results path."""
    runfiles = [Path(results_path, fname) for fname in os.listdir(results_path)
                if fname.startswith(f'{prefix}-')]
    run_results = [load_result(runfile) for runfile in runfiles]
    sizes = run_results[0][0]
    stacked = np.stack([values for _, values in run_results])
    average = np.average(stacked, axis=0)
    print(average.shape)
    return sizes, average


def choose_x_label(size):
    """Choose the proper x size label."""
    if size >= 1024:
        size_k = int(size / 1024)
        return f'{size_k}k'
    return size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-path', required=True, help='result path')
    parser.add_argument('-t', '--title', required=True, help='graph title')
    parser.add_argument('-b', '--benchmark', required=True, choices=('bw', 'latency'), help='benchmark type')
    args = parser.parse_args()

    kinds = {
        'custom': '-r',
        'packed': ':g',
        'iovec': '.-k',
        'rsmpi': '-^b',
    }
    labels = {
        'custom': 'mpicd-custom',
        'packed': 'mpicd-packed',
        'iovec': 'mpicd-regions',
        'rsmpi': 'rsmpi-bytes-baseline'
    }

    plt.style.use('paper.mplstyle')
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    for name, fmt in kinds.items():
        sizes, values = load_average_results(args.results_path, name)
        sizes = [choose_x_label(size) for size in sizes]
        ax.plot(sizes, values, fmt, label=labels[name])
    ax.set_title(args.title)
    ax.set_xlabel('size (bytes)')
    if args.benchmark == 'bw':
        ax.set_ylabel('bandwidth (MB/s)')
    else:
        ax.set_ylabel('latency (µs)')
    ax.legend()
    plt.show()
