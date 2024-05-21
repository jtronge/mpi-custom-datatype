use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd::datatype::MessageBuffer;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, ComplexVec, LatencyBenchmark, LatencyOptions,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    single_vec: bool,
    ctx: C,
    rank: i32,
    sbuf: Option<ComplexVec>,
    rbuf: Option<ComplexVec>,
}

unsafe fn inner_code<C: Communicator, S: MessageBuffer, R: MessageBuffer>(ctx: &C, rank: i32, sbuf: S, rbuf: R) {
    if rank == 0 {
        let sreq = ctx.isend(sbuf, 1, 0).expect("failed to send buffer to rank 1");
        let _ = ctx.waitall(&[sreq]);
        let rreq = ctx.irecv(rbuf, 1, 0).expect("failed to receive buffer from rank 1");
        let _ = ctx.waitall(&[rreq]);
    } else {
        let rreq = ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0");
        let _ = ctx.waitall(&[rreq]);
        let sreq = ctx.isend(sbuf, 0, 0).expect("failed to send buffer to rank 0");
        let _ = ctx.waitall(&[sreq]);
    }
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        let count = size / std::mem::size_of::<i32>();
        if self.single_vec {
            let _ = self.sbuf.insert(ComplexVec::single_vec(count));
            let _ = self.rbuf.insert(ComplexVec::single_vec(count));
        } else {
            let _ = self.sbuf.insert(ComplexVec::new(count, 2333));
            let _ = self.rbuf.insert(ComplexVec::new(count, 3222));
        }
        assert_eq!(
            self.rbuf
                .as_ref()
                .expect("missing buffer")
                .0
                .iter()
                .map(|v| v.len())
                .sum::<usize>() * std::mem::size_of::<i32>(),
            size,
        );
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
        single_vec: args.single_vec,
        ctx,
        rank,
        sbuf: None,
        rbuf: None,
    };
    mpicd_rust_benchmarks::latency(opts, benchmark, rank);
}
