#!/usr/bin/env python3
"""Plot the python pingpong results."""
import numpy as np
from matplotlib import pyplot as plt


def load_and_plot(fnames, title):
    results = {}
    for method in [
        'baseline',
        'pickle-oob-cdt',
        'pickle-oob',
        'pickle-basic',
    ]:
        size, data = np.genfromtxt(
            fnames[method],
            usecols=[0, 1],
            unpack=True
        )
        results[method] = [size, data]
    print(results)

    formats = {
        'baseline': '.-r',
        'pickle-oob-cdt': '+:b',
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
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    for name, (size, bw) in results.items():
        ax.plot(size, bw, formats[name], label=labels[name])
    ax.set_title(title)
    ax.set_xlabel('size (bytes)')
    ax.set_ylabel('bandwidth (MB/s)')
    ax.grid(which='both')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    load_and_plot({'baseline': 'results/python-pingpong/one-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong/one-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong/one-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong/one-node/pickle_basic.out'},
                  title='pingpong test (Python) - one node')
    load_and_plot({'baseline': 'results/python-pingpong/two-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong/two-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong/two-node/pickle_basic.out'},
                  title='pingpong test (Python) - two node')
    load_and_plot({'baseline': 'results/python-pingpong-complex-object-1mib/two-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong-complex-object-1mib/two-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong-complex-object-1mib/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong-complex-object-1mib/two-node/pickle_basic.out'},
                  title='pingpong test (Python) - complex object - 1 MiB buffers')
    load_and_plot({'baseline': 'results/python-pingpong-complex-object-16mib/two-node/baseline.out',
                   'pickle-oob-cdt': 'results/python-pingpong-complex-object-16mib/two-node/pickle_oob_cdt.out',
                   'pickle-oob': 'results/python-pingpong-complex-object-16mib/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong-complex-object-16mib/two-node/pickle_basic.out'},
                  title='pingpong test (Python) - complex object - 16 MiB buffers')
