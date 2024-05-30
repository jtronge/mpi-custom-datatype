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
            try:
                sizes.append(int(size))
            except ValueError as err:
                raise RuntimeError('ERROR:', size, fname) from None
            values.append(float(value))
    return np.array(sizes), np.array(values)


def load_average_results(results_path, prefix):
    """Load and average multiple runs in the results path."""
    runfiles = [Path(results_path, fname) for fname in os.listdir(results_path)
                if fname.startswith(f'{prefix}-')]
    run_results = [load_result(runfile) for runfile in runfiles]
    print(run_results)
    sizes = run_results[0][0]
    stacked = np.stack([values for _, values in run_results])
    average = np.average(stacked, axis=0)
    lower_error = average - np.min(stacked, axis=0)
    upper_error = np.max(stacked, axis=0) - average
    return sizes, average, (lower_error, upper_error)


def choose_x_label(size):
    """Choose the proper x size label."""
    if size >= (128 * 1024):
        size_m = size / 1024 / 1024
        return f'{size_m:.2}M'
    if size >= 1024:
        size_k = int(size / 1024)
        return f'{size_k}K'
    return str(size)


FMTS = {
    'custom': '-r',
    'packed': ':g',
    'rsmpi': '-^b',
    'rsmpi-struct-vec': '-.b',
}
LABELS = {
    'custom': 'custom',
    'packed': 'manual-pack',
    'rsmpi': 'rsmpi-bytes-baseline',
    'rsmpi-struct-vec': 'rsmpi-derived-datatype',
}

def basic(args):
    fig, ax = plt.subplots()
    # ax.axis('tight')

    all_values = []
    names = args.names.split(',')
    for name in names:
        sizes, values, err = load_average_results(args.results_path, name)
        all_values.append(values)
        ax.errorbar(x=sizes, y=values, yerr=err, fmt=FMTS[name], label=LABELS[name])

    # Determine bounds
    # ax.set_ybound(lower=args.lower, upper=args.upper)

    ax.set_title(args.title)
    ax.set_xlabel('size (bytes)')
    if args.benchmark == 'bw':
        ax.set_ylabel('bandwidth (MB/s)')
    else:
        ax.set_ylabel('latency (µs)')
    ax.grid(which='both')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    plt.show()


LATENCY_SIZE_COMPARE_RESULTS = [
    ('64 bytes', 'results/latency-64s', 'custom', '.-r'),
    ('128 bytes', 'results/latency-128s', 'custom', 's-.r'),
    ('512 bytes', 'results/latency-512s', 'custom', 'P--r'),
    ('1024 bytes', 'results/latency-1024s', 'custom', '*:r'),
    ('2048 bytes', 'results/latency-2048s', 'custom', 'D-.r'),
    ('rsmpi-bytes-baseline', 'results/latency-2048s', 'rsmpi', '-^b'),
    ('manual-pack', 'results/latency-2048s', 'packed', ':g'),
]

def latency_size_compare(args):
    fig, ax = plt.subplots()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    for label, path, name, fmt in LATENCY_SIZE_COMPARE_RESULTS:
        sizes, values, err = load_average_results(path, name)
        print(err)
        # sizes = [choose_x_label(size) for size in sizes]
        ax.errorbar(x=sizes, y=values, yerr=err, fmt=fmt, label=label)
    ax.set_title(args.title)
    ax.set_xlabel('size (bytes)')
    ax.set_ylabel('latency (µs)')
    ax.set_ybound(lower=1)
    ax.grid(which='both')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    plt.style.use('paper.mplstyle')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_basic = subparsers.add_parser('basic')
    parser_basic.add_argument('-r', '--results-path', required=True, help='result path')
    parser_basic.add_argument('-t', '--title', required=True, help='graph title')
    parser_basic.add_argument('-b', '--benchmark', required=True, choices=('bw', 'latency'), help='benchmark type')
    parser_basic.add_argument('-n', '--names', required=True, help='comma-separated list of benchmarks type names to plot')
    parser_basic.set_defaults(handler=basic)

    parser_latency_size_compare = subparsers.add_parser('latency-size-compare',
                                                        help='display graph for latency subvec size comparison tests')
    parser_latency_size_compare.add_argument('-t', '--title', required=True, help='graph title')
    parser_latency_size_compare.set_defaults(handler=latency_size_compare)

    args = parser.parse_args()
    args.handler(args)
