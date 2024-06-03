use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd::datatype::MessageBuffer;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, BenchmarkDatatype, ComplexVec, LatencyBenchmark,
    LatencyBenchmarkBuffer, LatencyOptions, ManualPack, StructVecArray, StructSimpleArray,
};

struct Benchmark<C: Communicator> {
    kind: BenchmarkKind,
    subvector_size: usize,
    ctx: C,
    rank: i32,
    buffers: LatencyBenchmarkBuffer,
}

unsafe fn inner_code<C: Communicator, B: MessageBuffer + ?Sized>(
    ctx: &C,
    rank: i32,
    sbuf: &B,
    rbuf: &mut B,
) {
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

fn latency<C: Communicator, B: MessageBuffer + ManualPack + ?Sized>(
    kind: BenchmarkKind,
    ctx: &C,
    rank: i32,
    sbuf: &B,
    rbuf: &mut B,
) {
    unsafe {
        match kind {
            BenchmarkKind::Packed => {
                let packed_sbuf = sbuf.manual_pack();
                let mut packed_rbuf = vec![0u8; packed_sbuf.len()];
                inner_code(ctx, rank, &packed_sbuf[..], &mut packed_rbuf[..]);
                rbuf.manual_unpack(&packed_rbuf);
            }
            BenchmarkKind::Custom => {
                inner_code(ctx, rank, sbuf, rbuf);
            }
        }
    }
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        match self.buffers {
            LatencyBenchmarkBuffer::DoubleVec(ref mut buffers) => {
                let count = size / std::mem::size_of::<i32>();
                let subvector_count = self.subvector_size / std::mem::size_of::<i32>();
                let _ = buffers.insert((ComplexVec::new(count, subvector_count), ComplexVec::new(count, subvector_count)));
                assert_eq!(
                    buffers
                        .as_ref()
                        .expect("missing buffer")
                        .0
                        .0
                        .iter()
                        .map(|v| v.len())
                        .sum::<usize>() * std::mem::size_of::<i32>(),
                    size,
                );
            }
            LatencyBenchmarkBuffer::StructVec(ref mut buffers) => {
                let _ = buffers.insert((StructVecArray::new(size), StructVecArray::new(size)));
            }
            LatencyBenchmarkBuffer::StructSimple(ref mut buffers) => {
                let _ = buffers.insert((StructSimpleArray::new(size), StructSimpleArray::new(size)));
            }
        }
    }

    fn body(&mut self) {
        match &mut self.buffers {
            LatencyBenchmarkBuffer::DoubleVec(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing buffers");
                latency(self.kind, &self.ctx, self.rank, sbuf, rbuf);
            }
            LatencyBenchmarkBuffer::StructVec(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing buffers");
                latency(self.kind, &self.ctx, self.rank, sbuf, rbuf);
            }
            LatencyBenchmarkBuffer::StructSimple(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing buffers");
                latency(self.kind, &self.ctx, self.rank, sbuf, rbuf);
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

    let buffers = match args.datatype {
        BenchmarkDatatype::DoubleVec => LatencyBenchmarkBuffer::DoubleVec(None),
        BenchmarkDatatype::StructVec => LatencyBenchmarkBuffer::StructVec(None),
        BenchmarkDatatype::StructSimple => LatencyBenchmarkBuffer::StructSimple(None),
    };
    let benchmark = Benchmark {
        kind: args.kind,
        subvector_size: args.subvector_size,
        ctx,
        rank,
        buffers,
    };
    mpicd_rust_benchmarks::latency(opts, benchmark, rank);
}
