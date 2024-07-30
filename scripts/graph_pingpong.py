#!/usr/bin/env python3
"""Plot the python pingpong results."""
import numpy as np
from matplotlib import pyplot as plt


def load_and_plot(fnames, title, x_start=0, x_label='size (bytes)'):
    results = {}
    for method in [
        'baseline',
        'pickle-oob-cdt',
        'pickle-oob',
        'pickle-basic',
    ]:
        size, bw, lat, std_dev = np.genfromtxt(
            fnames[method],
            usecols=[0, 1, 3, 5],
            unpack=True
        )
        results[method] = [size, bw, lat, std_dev]
    print(results)

    formats = {
        'baseline': '+:k',
        'pickle-oob-cdt': 'o-r',
        'pickle-oob': '.-.g',
        'pickle-basic': '>-b',
    }

    labels = {
        'baseline': 'baseline',
        'pickle-oob-cdt': 'pickle-oob-cdt',
        'pickle-oob': 'pickle-oob',
        'pickle-basic': 'pickle-basic',
    }

    plt.style.use('paper.mplstyle')

    fig, ax = plt.subplots()
    for name, (size, bw, lat, std_dev) in results.items():
        size_mb = size / 1000**2
        max_bw = size_mb / (lat - std_dev)
        yerr = max_bw - bw
        ax.errorbar(size[x_start:], bw[x_start:], yerr=yerr[x_start:],
                    fmt=formats[name], label=labels[name])
    ax.set_title(title)
    ax.set_xlabel('size (bytes)')
    ax.set_ylabel('bandwidth (MB/s)')
    ax.grid(which='both')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    load_and_plot({'baseline': 'results/python-pingpong-complex-object-128kib/two-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong-complex-object-128kib/two-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong-complex-object-128kib/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong-complex-object-128kib/two-node/pickle_basic.out'},
                  title='pingpong test (Python) - complex object', x_label='total size (bytes)')
    load_and_plot({'baseline': 'results/python-pingpong/two-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong/two-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong/two-node/pickle_basic.out'},
                  title='pingpong test (Python)', x_start=15)
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-1mib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-1mib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-1mib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-1mib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 1 MiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-2mib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-2mib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-2mib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-2mib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 2 MiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-8mib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-8mib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-8mib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-8mib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 8 MiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-16mib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-16mib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-16mib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-16mib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 16 MiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-64kib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-64kib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-64kib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-64kib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 64 KiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-256kib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-256kib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-256kib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-256kib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 256 KiB buffers')
    # load_and_plot({'baseline': 'results/python-pingpong-complex-object-512kib/two-node/baseline.out',
    #                'pickle-oob-cdt': 'results/python-pingpong-complex-object-512kib/two-node/pickle_oob_cdt.out',
    #                'pickle-oob': 'results/python-pingpong-complex-object-512kib/two-node/pickle_oob.out',
    #                'pickle-basic': 'results/python-pingpong-complex-object-512kib/two-node/pickle_basic.out'},
    #               title='pingpong test (Python) - complex object - 512 KiB buffers')
