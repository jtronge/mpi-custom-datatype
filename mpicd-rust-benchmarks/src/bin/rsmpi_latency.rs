use clap::Parser;
use mpicd_rust_benchmarks::{RsmpiArgs, BenchmarkDatatype, LatencyBenchmark, LatencyOptions};
use mpi::traits::*;

struct Benchmark<C: Communicator> {
    comm: C,
    rank: i32,
    sbuf: Option<Vec<u8>>,
    rbuf: Option<Vec<u8>>,
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        let _ = self.sbuf.insert(vec![0; size]);
        let _ = self.rbuf.insert(vec![0; size]);
    }

    fn body(&mut self) {
        let next_rank = (self.rank + 1) % 2;
        let proc = self.comm.process_at_rank(next_rank);
        let sbuf = self.sbuf.as_ref().expect("missing send buffer");
        let rbuf = self.rbuf.as_mut().expect("missing receive buffer");

        if self.rank == 0 {
            proc.send(sbuf);
            let _ = proc.receive_into(rbuf);
        } else {
            let _ = proc.receive_into(rbuf);
            proc.send(sbuf);
        }
    }
}

fn main() {
    let args = RsmpiArgs::parse();
    let opts: LatencyOptions = mpicd_rust_benchmarks::load_options(args.options_path);
    let universe = mpi::initialize().expect("failed to initialize rsmpi");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    assert_eq!(size, 2);

    let benchmark = Benchmark {
        comm: world,
        rank,
        sbuf: None,
        rbuf: None,
    };
    mpicd_rust_benchmarks::latency(opts, benchmark, rank);
}
