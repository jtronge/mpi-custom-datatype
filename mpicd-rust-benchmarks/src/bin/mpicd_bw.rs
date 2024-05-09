use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, BandwidthOptions, BandwidthBenchmark, ComplexVec,
    generate_complex_vec,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    ctx: C,
    rank: i32,
    buffers: Option<Vec<ComplexVec>>,
}

impl<C: Communicator> BandwidthBenchmark for Benchmark<C> {
    fn init(&mut self, window_size: usize, size: usize) {
        let buffers = if self.rank == 0 {
            (0..window_size).map(|i| generate_complex_vec(size, i+1)).collect()
        } else {
            (0..window_size).map(|_| ComplexVec(vec![vec![0; size]])).collect()
        };
        let _ = self.buffers.insert(buffers);
    }

    fn body(&mut self) {
        let buffers = self.buffers.as_mut().expect("missing buffers");
        unsafe {
            if self.rank == 0 {
                let mut reqs = vec![];
                for sbuf in buffers {
                    match self.kind {
                        BenchmarkKind::Packed => {
                            let packed_sbuf = sbuf.pack();
                            reqs.push(self.ctx.isend(&packed_sbuf[..], 1, 0).expect("failed to send buffer to rank 1"));
                        }
                        BenchmarkKind::Custom => {
                            reqs.push(self.ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1"));
                        }
                    }
                }
                let _ = self.ctx.waitall(&reqs);

                let mut ack_buf = ComplexVec(vec![vec![0]; 1]);
                let ack_req = self.ctx.irecv(&mut ack_buf, 1, 0).expect("failed to receive ack buf from rank 1");
                let _ = self.ctx.waitall(&[ack_req]);
                assert_eq!(ack_buf.0[0][0], 2);
            } else {
                match self.kind {
                    BenchmarkKind::Packed => {
                        // Receive the buffers packed.
                        let mut reqs = vec![];
                        let mut packed_rbufs = buffers.iter().map(|buf| buf.pack()).collect::<Vec<Vec<i32>>>();
                        for packed_rbuf in &mut packed_rbufs {
                            reqs.push(self.ctx.irecv(&mut packed_rbuf[..], 0, 0).expect("failed to receive packed buffer from rank 0"));
                        }
                        let _ = self.ctx.waitall(&reqs);

                        // Unpack the buffers into the ComplexVec types.
                        for (buffer, packed_rbuf) in buffers.iter_mut().zip(packed_rbufs) {
                            buffer.unpack_from(&packed_rbuf);
                        }
                    }
                    BenchmarkKind::Custom => {
                        let mut reqs = vec![];
                        for rbuf in buffers {
                            reqs.push(self.ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0"));
                        }
                        let _ = self.ctx.waitall(&reqs);
                    }
                }

                let ack_buf = ComplexVec(vec![vec![2]; 1]);
                let ack_req = self.ctx.isend(&ack_buf, 0, 0).expect("failed to send ack to rank 0");
                let _ = self.ctx.waitall(&[ack_req]);
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

    let opts = BandwidthOptions {
        min_size: 8,
        max_size: 1024,
        window_size: 64,
        iterations: 1024,
        skip: 10,
        warmup_validation: 20,
    };
    let benchmark = Benchmark {
        kind: args.kind,
        ctx,
        rank,
        buffers: None,
    };
    let results = mpicd_rust_benchmarks::bw(opts, benchmark);

    if rank == 0 {
        println!("# size bandwidth");
        for (size, bw) in &results {
            println!("{} {}", size, bw);
        }
    }
}
