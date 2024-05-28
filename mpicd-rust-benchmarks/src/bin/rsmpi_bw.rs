use clap::Parser;
use mpi::traits::*;
use mpicd_rust_benchmarks::{RsmpiArgs, BenchmarkDatatype, BandwidthOptions, BandwidthBenchmark};

struct RsmpiBenchmark<C: Communicator> {
    datatype: BenchmarkDatatype,
    comm: C,
    rank: i32,
    sbuf: Option<Vec<Vec<u8>>>,
    rbuf: Option<Vec<Vec<u8>>>,
}

impl<C: Communicator> BandwidthBenchmark for RsmpiBenchmark<C> {
    fn init(&mut self, window_size: usize, size: usize) {
        let sbuf = (0..window_size).map(|_| vec![0u8; size]).collect();
        let _ = self.sbuf.insert(sbuf);
        let rbuf = (0..window_size).map(|_| vec![0u8; size]).collect();
        let _ = self.rbuf.insert(rbuf);
    }

    fn body(&mut self) {
        let sbuf = self.sbuf.as_ref().unwrap();
        let rbuf = self.rbuf.as_mut().unwrap();
        let window_size = sbuf.len();
        let proc = self.comm.process_at_rank((self.rank + 1) % 2);
        mpi::request::multiple_scope(window_size, |scope, coll| {
            if self.rank == 0 {
                for i in 0..window_size {
                    coll.add(proc.immediate_send(scope, &sbuf[i]));
                }
            } else {
                let mut tmp = &mut rbuf[..];
                for _ in 0..window_size {
                    let (first, second) = tmp.split_at_mut(1);
                    tmp = second;
                    coll.add(proc.immediate_receive_into(scope, &mut first[0]));
                }
            }
            let mut stats = vec![];
            coll.wait_all(&mut stats);
        });

        if self.rank == 0 {
            let (_, _): (Vec<i32>, _) = proc.receive_vec();
        } else {
            proc.send(&[0i32]);
        }
    }
}

fn main() {
    let args = RsmpiArgs::parse();
    let opts: BandwidthOptions = mpicd_rust_benchmarks::load_options(&args.options_path);
    let universe = mpi::initialize().expect("failed to initialize rsmpi");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    assert_eq!(size, 2);

    let benchmark = RsmpiBenchmark {
        datatype: args.datatype,
        comm: world,
        rank,
        sbuf: None,
        rbuf: None,
    };
    mpicd_rust_benchmarks::bw(opts, benchmark, rank);
}
