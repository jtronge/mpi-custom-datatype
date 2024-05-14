use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd::datatype::Buffer;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, ComplexVec, IovecComplexVec, IovecComplexVecMut, latency, LatencyBenchmark, LatencyOptions, generate_complex_vec,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    ctx: C,
    rank: i32,
    sbuf: Option<ComplexVec>,
    rbuf: Option<ComplexVec>,
}

unsafe fn inner_code<C: Communicator, S: Buffer, R: Buffer>(ctx: &C, rank: i32, sbuf: S, rbuf: R) {
    if rank == 0 {
        let sreq = ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1");
        let rreq = ctx.irecv(rbuf, 1, 0).expect("failed to receive buffer from rank 1");
        let _ = ctx.waitall(&[sreq, rreq]);
    } else {
        let rreq = ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0");
        let sreq = ctx.isend(sbuf, 0, 0).expect("failed to send buffer to rank 0");
        let _ = ctx.waitall(&[sreq, rreq]);
    }
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        let true_size = size / std::mem::size_of::<i32>();
        let _ = self.sbuf.insert(generate_complex_vec(true_size, 2333));
        let _ = self.rbuf.insert(ComplexVec(vec![vec![0; true_size]]));
    }

    fn body(&mut self) {
        let sbuf = self.sbuf.as_ref().expect("missing send buffer");
        let rbuf = self.rbuf.as_mut().expect("missing buffer");

        unsafe {
            match self.kind {
                BenchmarkKind::Packed => {
                    let packed_sbuf = sbuf.pack();
                    let mut packed_rbuf = vec![0i32; packed_sbuf.len()];
                    inner_code(&self.ctx, self.rank, &packed_sbuf[..], &mut packed_rbuf[..]);
                    rbuf.unpack_from(&packed_rbuf);
                }
                BenchmarkKind::Custom => {
                    inner_code(&self.ctx, self.rank, sbuf, rbuf);
                }
                BenchmarkKind::Iovec => {
                    let sbuf_iov = IovecComplexVec(sbuf);
                    let rbuf_iov = IovecComplexVecMut(rbuf);
                    inner_code(&self.ctx, self.rank, sbuf_iov, rbuf_iov);
                }
            }
        }
    }
}

fn main() {
    let args = BenchmarkArgs::parse();
    let opts: LatencyOptions = mpicd_rust_benchmarks::load_options(&args.options_path);

    let ctx = mpicd::init().expect("failed to init mpicd");
    let size = ctx.size();
    let rank = ctx.rank();
    assert_eq!(size, 2);

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
