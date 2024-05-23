use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, BandwidthOptions, BandwidthBenchmark, ComplexVec,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    ctx: C,
    rank: i32,
    subvector_size: usize,
    buffers: Option<Vec<ComplexVec>>,
}

impl<C: Communicator> BandwidthBenchmark for Benchmark<C> {
    fn init(&mut self, window_size: usize, size: usize) {
        let count = size / std::mem::size_of::<i32>();
        let subvector_count = self.subvector_size / std::mem::size_of::<i32>();

        if let Some(buffers) = self.buffers.as_mut() {
            // Assume the window size doesn't change between iterations.
            assert_eq!(buffers.len(), window_size);

            for (i, buf) in buffers.iter_mut().enumerate() {
                buf.update(count, subvector_count);
            }
        } else {
            let buffers = (0..window_size)
                .map(|i| ComplexVec::new(count, subvector_count))
                .collect();
            let _ = self.buffers.insert(buffers);
        }
    }

    fn body(&mut self) {
        let buffers = self.buffers.as_mut().expect("missing buffers");
        unsafe {
            if self.rank == 0 {
                let mut reqs = vec![];
                for sbuf in buffers {
                    let req = match self.kind {
                        BenchmarkKind::Packed => {
                            let packed_sbuf = sbuf.pack();
                            self.ctx.isend(&packed_sbuf[..], 1, 0)
                        }
                        BenchmarkKind::Custom => {
                            self.ctx.isend(sbuf, 1, 0)
                        }
                    };
                    reqs.push(req.expect("failed to send buffer to rank 1"));
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
    let opts: BandwidthOptions = mpicd_rust_benchmarks::load_options(&args.options_path);

    let ctx = mpicd::init().expect("failed to init mpicd");
    let size = ctx.size();
    let rank = ctx.rank();
    assert_eq!(size, 2);

    let benchmark = Benchmark {
        kind: args.kind,
        ctx,
        rank,
        subvector_size: args.subvector_size,
        buffers: None,
    };
    mpicd_rust_benchmarks::bw(opts, benchmark, rank);
}
