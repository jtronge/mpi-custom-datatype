use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{BandwidthOptions, BandwidthBenchmark, ComplexVec, generate_complex_vec};

struct Benchmark<C: Communicator> {
    ctx: C,
    size: i32,
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
        unsafe {
            if self.rank == 0 {
                let buffers = self.buffers.as_ref().expect("missing buffers");
                let mut reqs = vec![];
                for sbuf in buffers {
                    reqs.push(self.ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1"));
                }
                let _ = self.ctx.waitall(&reqs);

                let mut ack_buf = ComplexVec(vec![vec![0]; 1]);
                let ack_req = self.ctx.irecv(&mut ack_buf, 1, 0).expect("failed to receive ack buf from rank 1");
                let _ = self.ctx.waitall(&[ack_req]);
                assert_eq!(ack_buf.0[0][0], 2);
            } else {
                let buffers = self.buffers.as_mut().expect("missing buffers");
                let mut reqs = vec![];
                for rbuf in buffers {
                    reqs.push(self.ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0"));
                }
                let _ = self.ctx.waitall(&reqs);

                let ack_buf = ComplexVec(vec![vec![2]; 1]);
                let ack_req = self.ctx.isend(&ack_buf, 0, 0).expect("failed to send ack to rank 0");
                let _ = self.ctx.waitall(&[ack_req]);
            }
        }
    }
}

fn main() {
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
        ctx,
        size,
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
