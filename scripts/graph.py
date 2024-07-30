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
    'custom': 's-r',
    'packed': '>:g',
    'rsmpi': 'H-.b',
    'rsmpi-struct-vec': 'H-.b',
    'rsmpi-struct-simple': 'H-.b',
    'rsmpi-struct-simple-no-gap': 'H-.b',
}
LABELS = {
    'custom': 'custom',
    'packed': 'manual-pack',
    'rsmpi': 'rsmpi-bytes-baseline',
    'rsmpi-struct-vec': 'rsmpi-derived-datatype',
    'rsmpi-struct-simple': 'rsmpi-derived-datatype',
    'rsmpi-struct-simple-no-gap': 'rsmpi-derived-datatype',
}

def basic_graph(results_path, names, title, benchmark, y_bounds=None, max_x=None):
    fig, ax = plt.subplots()

    for name in names:
        sizes, values, err = load_average_results(results_path, name)
        # Graph up to max_x value, if specified.
        if max_x is not None:
            sizes = [sz for sz in sizes if sz <= max_x]
            print(len(sizes))
            values = values[:len(sizes)]
            print(values.shape)
            err = err[0][:len(sizes)], err[1][:len(sizes)]
        ax.errorbar(x=sizes, y=values, yerr=err, fmt=FMTS[name], label=LABELS[name])

    ax.set_title(title)
    ax.set_xlabel('size (bytes)')
    if benchmark == 'bw':
        ax.set_ylabel('bandwidth (MB/s)')
    else:
        ax.set_ylabel('latency (µs)')
    ax.grid(which='both')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    # Determine bounds
    if y_bounds is not None:
        ax.set_ybound(lower=y_bounds[0], upper=y_bounds[1])
    plt.show()


def bw_double_vec(_args):
    """Graph bandwidth for the double-vec type."""
    basic_graph('results/bw_double-vec_1024s', ['custom', 'packed', 'rsmpi'],
                'Bandwidth double-vec (subvector 1024 bytes)', 'bw')


LATENCY_SIZE_COMPARE_RESULTS = [
    ('custom-64b', 'results/latency_double-vec/custom-64', '.-r'),
    ('custom-256b', 'results/latency_double-vec/custom-256', 's-r'),
    ('custom-1k', 'results/latency_double-vec/custom-1024', 'P-r'),
    ('custom-4k', 'results/latency_double-vec/custom-4096', '*-r'),
    ('manual-pack-64b', 'results/latency_double-vec/packed-64', '>:g'),
    ('manual-pack-256b', 'results/latency_double-vec/packed-256', '<:g'),
    ('manual-pack-1k', 'results/latency_double-vec/packed-1024', '+:g'),
    ('manual-pack-4k', 'results/latency_double-vec/packed-4096', '3:g'),
    ('rsmpi-bytes-baseline', 'results/latency_double-vec/rsmpi-bytes', 'H-.b'),
]

def latency_double_vec(_args):
    fig, ax = plt.subplots()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    for label, path, fmt in LATENCY_SIZE_COMPARE_RESULTS:
        sizes, values, err = load_average_results(path, 'result')
        ax.errorbar(x=sizes, y=values, yerr=err, fmt=fmt, label=label)
    ax.set_title('Latency Varying Subvector Sizes')
    ax.set_xlabel('size (bytes)')
    ax.set_ylabel('latency (µs)')
    ax.set_ybound(lower=1)
    ax.grid(which='both')
    ax.legend()
    plt.show()


def bw_struct_vec(_args):
    """Graph banwdith for the struct-vec type."""
    basic_graph('results/bw_struct-vec', ['custom', 'packed', 'rsmpi-struct-vec'],
                'Bandwidth (struct-vec)', 'bw', max_x=2**20)


def latency_struct_vec(_args):
    """Graph latency for the struct-vec type."""
    basic_graph('results/latency_struct-vec', ['custom', 'packed', 'rsmpi-struct-vec'],
                'Latency (struct-vec)', 'latency')


def bw_struct_simple(_args):
    """Graph banwdith for the struct-simple type."""
    basic_graph('results/bw_struct-simple', ['custom', 'packed', 'rsmpi-struct-simple'],
                'Bandwidth (struct-simple)', 'bw')


def latency_struct_simple(_args):
    """Graph latency for the struct-simple type."""
    basic_graph('results/latency_struct-simple', ['custom', 'packed', 'rsmpi-struct-simple'],
                'Latency (struct-simple)', 'latency')


def latency_struct_simple_no_gap(_args):
    """Graph latency for the struct-simple-no-gap type."""
    basic_graph('results/latency_struct-simple-no-gap',
                ['custom', 'packed', 'rsmpi-struct-simple-no-gap'],
                'Latency (struct-simple-no-gap)', 'latency')


if __name__ == '__main__':
    plt.style.use('paper.mplstyle')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser('bw-double-vec').set_defaults(handler=bw_double_vec)
    subparsers.add_parser('latency-double-vec').set_defaults(handler=latency_double_vec)
    subparsers.add_parser('bw-struct-vec').set_defaults(handler=bw_struct_vec)
    subparsers.add_parser('latency-struct-vec').set_defaults(handler=latency_struct_vec)
    subparsers.add_parser('bw-struct-simple').set_defaults(handler=bw_struct_simple)
    subparsers.add_parser('latency-struct-simple').set_defaults(handler=latency_struct_simple)
    subparsers.add_parser('latency-struct-simple-no-gap').set_defaults(handler=latency_struct_simple_no_gap)
    args = parser.parse_args()
    args.handler(args)
