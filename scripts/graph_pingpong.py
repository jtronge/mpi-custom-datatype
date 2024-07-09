#!/usr/bin/env python3
"""Plot the python pingpong results."""
from matplotlib import pyplot as plt


def load_and_plot(fnames, title):
    results = {}
    for method in ['baseline', 'pickle-oob', 'pickle-basic']:
        data = []
        size = []
        with open(fnames[method]) as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                line = line.split()
                sz = int(line[0])
                bw = float(line[1])
                size.append(sz)
                data.append(bw)
        results[method] = [size, data]
    print(results)

    formats = {
        'baseline': '.-r',
        'pickle-oob': '+:b',
        'pickle-basic': '>-b',
    }

    labels = {
        'baseline': 'baseline',
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
                   'pickle-oob': 'results/python-pingpong/one-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong/one-node/pickle_basic.out'},
                  title='pingpong test (Python) - one node')
    load_and_plot({'baseline': 'results/python-pingpong/two-node/baseline.out',
                   'pickle-oob': 'results/python-pingpong/two-node/pickle_oob.out',
                   'pickle-basic': 'results/python-pingpong/two-node/pickle_basic.out'},
                  title='pingpong test (Python) - two node')
