# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
"""Run MPI benchmarks and tests."""
import time


def pingpong(MPI, args=None, verbose=True):
    """Time messages between processes."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=import-outside-toplevel
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="pingpong")
    parser.add_argument("-q", "--quiet", action="store_false",
                        dest="verbose", default=verbose,
                        help="quiet output")
    parser.add_argument("-m", "--min-size", type=int,
                        dest="min_size", default=1,
                        help="minimum message size")
    parser.add_argument("-n", "--max-size", type=int,
                        dest="max_size", default=1 << 30,
                        help="maximum message size")
    parser.add_argument("-s", "--skip", type=int,
                        dest="skip", default=100,
                        help="number of warm-up iterations")
    parser.add_argument("-l", "--loop", type=int,
                        dest="loop", default=10000,
                        help="number of iterations")
    parser.add_argument("-a", "--array", action="store",
                        dest="array", default="numpy",
                        choices=["numpy", "array", "none"],
                        help="use NumPy/array arrays")
    parser.add_argument("-p", "--pickle", action="store_true",
                        dest="pickle", default=False,
                        help="use pickle-based send and receive")
    #parser.add_argument("--protocol", type=int,
    #                    dest="protocol", default=None,
    #                    help="pickle protocol version")
    parser.add_argument("-o", "--outband", action="store_true",
                        dest="outband", default=False,
                        help="use out-of-band pickle-based send and receive")
    parser.add_argument("--skip-large", type=int,
                        dest="skip_large", default=10)
    parser.add_argument("--loop-large", type=int,
                        dest="loop_large", default=1000)
    parser.add_argument("--large-size", type=int,
                        dest="large_size", default=1 << 14)
    parser.add_argument("--skip-huge", type=int,
                        dest="skip_huge", default=1)
    parser.add_argument("--loop-huge", type=int,
                        dest="loop_huge", default=10)
    parser.add_argument("--huge-size", type=int,
                        dest="huge_size", default=1 << 20)
    parser.add_argument("--no-header", action="store_false",
                        dest="print_header", default=True)
    parser.add_argument("--no-stats", action="store_false",
                        dest="print_stats", default=True)
    options = parser.parse_args(args)

    import statistics

    # pylint: disable=import-error
    numpy = array = None
    if options.array == 'numpy':
        import numpy
    elif options.array == 'array':
        import array

    skip = options.skip
    loop = options.loop
    min_size = options.min_size
    max_size = options.max_size
    skip_large = options.skip_large
    loop_large = options.loop_large
    large_size = options.large_size
    skip_huge = options.skip_huge
    loop_huge = options.loop_huge
    huge_size = options.huge_size

    use_pickle = options.pickle or options.outband
    use_outband = options.outband

    buf_sizes = [1 << i for i in range(33)]
    buf_sizes = [n for n in buf_sizes if min_size <= n <= max_size]

    wtime = time.perf_counter
    comm = MPI.COMM_WORLD
    if use_outband:
        send = comm.send_oob
        recv = comm.recv_oob
    elif use_pickle:
        send = comm.send
        recv = comm.recv
    else:
        send = comm.Send
        recv = comm.Recv
    s_msg = r_msg = None

    def allocate(nbytes):  # pragma: no cover
        if numpy:
            return numpy.empty(nbytes, 'B')
        elif array:
            return array.array('B', [0]) * nbytes
        else:
            return bytearray(nbytes)

    def run_pingpong():
        rank = comm.Get_rank()
        size = comm.Get_size()
        t_start = wtime()
        if size == 1:
            pass
        elif rank == 0:
            send(s_msg, 1, 0)
            recv(r_msg, 1, 0)
        elif rank == 1:
            recv(r_msg, 0, 0)
            send(s_msg, 0, 0)
        t_end = wtime()
        return (t_end - t_start) / 2

    result = []
    for nbytes in buf_sizes:
        if nbytes > large_size:
            skip = min(skip, skip_large)
            loop = min(loop, loop_large)
        if nbytes > huge_size:
            skip = min(skip, skip_huge)
            loop = min(loop, loop_huge)
        iterations = list(range(loop + skip))

        if use_pickle or use_outband:
            s_msg = allocate(nbytes)
        else:
            s_msg = [allocate(nbytes), nbytes, MPI.BYTE]
            r_msg = [allocate(nbytes), nbytes, MPI.BYTE]

        t_list = []
        comm.barrier()
        for i in iterations:
            elapsed = run_pingpong()
            if i >= skip:
                t_list.append(elapsed)

        s_msg = r_msg = None

        t_mean = statistics.mean(t_list) if t_list else float('nan')
        t_stdev = statistics.stdev(t_list) if len(t_list) > 1 else 0.0
        result.append((nbytes, t_mean, t_stdev))

        if options.verbose and comm.rank == 0:
            if options.print_header:
                options.print_header = False
                print("# MPI PingPong Test")
                header = "# Size [B]  Bandwidth [MB/s]"
                if options.print_stats:
                    header += " | Time Mean [s] \u00b1 StdDev [s]  Samples"
                print(header, flush=True)
            bandwidth = nbytes / t_mean
            message = f"{nbytes:10d}{bandwidth / 1e6:18.2f}"
            if options.print_stats:
                message += f" | {t_mean:.7e} \u00b1 {t_stdev:.4e} {loop:8d}"
            print(message, flush=True)

    return result


if __name__ == '__main__':
    # import mpi4py.MPI as MPI
    import mpicd as MPI
    pingpong(MPI)
