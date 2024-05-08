use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{ComplexVec, latency, LatencyBenchmark, LatencyOptions, generate_complex_vec};

struct Benchmark<C: Communicator> {
    ctx: C,
    size: i32,
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
        unsafe {
            if self.rank == 0 {
                let sbuf = self.sbuf.as_ref().expect("missing send buffer");
                let sreq = self.ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1");
                let rbuf = self.rbuf.as_mut().expect("missing buffer");
                let rreq = self.ctx.irecv(rbuf, 1, 0).expect("failed to receive buffer from rank 1");
                let _ = self.ctx.waitall(&[sreq, rreq]);
            } else {
                let rbuf = self.rbuf.as_mut().expect("missing buffer");
                let rreq = self.ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0");
                let sbuf = self.sbuf.as_ref().expect("missing send buffer");
                let sreq = self.ctx.isend(sbuf, 0, 0).expect("failed to send buffer to rank 0");
                let _ = self.ctx.waitall(&[sreq, rreq]);
            }
        }
    }
}

fn main() {
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
        ctx,
        size,
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
