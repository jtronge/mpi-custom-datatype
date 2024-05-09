use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, ComplexVec, latency, LatencyBenchmark, LatencyOptions, generate_complex_vec,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    ctx: C,
    rank: i32,
    sbuf: Option<ComplexVec>,
    rbuf: Option<ComplexVec>,
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        let _ = self.sbuf.insert(generate_complex_vec(size, 2333));
        let _ = self.rbuf.insert(ComplexVec(vec![vec![0; size]]));
    }

    fn body(&mut self) {
        let sbuf = self.sbuf.as_ref().expect("missing send buffer");
        let rbuf = self.rbuf.as_mut().expect("missing buffer");

        unsafe {
            match self.kind {
                BenchmarkKind::Packed => {
                    let packed_sbuf = sbuf.pack();
                    let mut packed_rbuf = vec![0i32; packed_sbuf.len()];

                    if self.rank == 0 {
                        let sreq = self.ctx.isend(&packed_sbuf[..], 1, 0).expect("failed to send buffer to rank 1");
                        let rreq = self.ctx.irecv(&mut packed_rbuf[..], 1, 0).expect("failed to send buffer to rank 1");
                        let _ = self.ctx.waitall(&[sreq, rreq]);
                    } else {
                        let rreq = self.ctx.irecv(&mut packed_rbuf[..], 0, 0).expect("failed to receive buffer from rank 0");
                        let sreq = self.ctx.isend(&packed_sbuf[..], 0, 0).expect("failed to send buffer to rank 0");
                        let _ = self.ctx.waitall(&[sreq, rreq]);
                    }

                    rbuf.unpack_from(&packed_rbuf);
                }
                BenchmarkKind::Custom => {
                    if self.rank == 0 {
                        let sreq = self.ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1");
                        let rreq = self.ctx.irecv(rbuf, 1, 0).expect("failed to receive buffer from rank 1");
                        let _ = self.ctx.waitall(&[sreq, rreq]);
                    } else {
                        let rreq = self.ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0");
                        let sreq = self.ctx.isend(sbuf, 0, 0).expect("failed to send buffer to rank 0");
                        let _ = self.ctx.waitall(&[sreq, rreq]);
                    }
                }
            }
        }
    }
}

fn main() {
    let args = BenchmarkArgs::parse();

    let ctx = mpicd::init().expect("failed to init mpicd");
    let size = ctx.size();
    let rank = ctx.rank();
    assert_eq!(size, 2);

    let opts = LatencyOptions {
        iterations: 100,
        skip: 10,
        warmup_validation: 10,
        min_size: 8,
        max_size: 100000,
    };
    let benchmark = Benchmark {
        kind: args.kind,
        ctx,
        rank,
        sbuf: None,
        rbuf: None,
    };
    let results = latency(opts, benchmark);

    if rank == 0 {
        println!("# size latency");
        for (size, lat) in &results {
            println!("{} {}", size, lat);
        }
    }
}
