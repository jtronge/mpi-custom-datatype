use clap::Parser;
use mpicd::communicator::Communicator;
use mpicd::datatype::MessageBuffer;
use mpicd_rust_benchmarks::{
    BenchmarkArgs, BenchmarkKind, BenchmarkDatatype, BenchmarkDatatypeBuffer,
    BandwidthOptions, BandwidthBenchmark, ManualPack, ComplexVec, StructVecArray,
    STRUCT_VEC_PACKED_SIZE_TOTAL,
};

struct Benchmark<R, C>
where
    C: Communicator<Request = R>,
{
    kind: BenchmarkKind,
    ctx: C,
    rank: i32,
    subvector_size: usize,
    buffers: BenchmarkDatatypeBuffer,
}

fn bandwidth<R, C, B>(
    rank: i32,
    ctx: &C,
    kind: BenchmarkKind,
    buffers: &mut [B],
)
where
    C: Communicator<Request = R>,
    B: MessageBuffer + ManualPack,
{
    unsafe {
        if rank == 0 {
            let mut reqs = vec![];
            for sbuf in buffers {
                let req = match kind {
                    BenchmarkKind::Packed => {
                        let packed_sbuf = sbuf.manual_pack();
                        ctx.isend(&packed_sbuf[..], 1, 0)
                    }
                    BenchmarkKind::Custom => {
                        ctx.isend(sbuf, 1, 0)
                    }
                };
                reqs.push(req.expect("failed to send buffer to rank 1"));
            }
            let _ = ctx.waitall(&reqs);

            let mut ack_buf = ComplexVec(vec![vec![0]; 1]);
            let ack_req = ctx.irecv(&mut ack_buf, 1, 0).expect("failed to receive ack buf from rank 1");
            let _ = ctx.waitall(&[ack_req]);
            assert_eq!(ack_buf.0[0][0], 2);
        } else {
            match kind {
                BenchmarkKind::Packed => {
                    // Receive the buffers packed.
                    let mut reqs = vec![];
                    let mut packed_rbufs = buffers.iter().map(|buf| buf.manual_pack()).collect::<Vec<Vec<u8>>>();
                    for packed_rbuf in &mut packed_rbufs {
                        reqs.push(ctx.irecv(&mut packed_rbuf[..], 0, 0).expect("failed to receive packed buffer from rank 0"));
                    }
                    let _ = ctx.waitall(&reqs);

                    // Unpack the buffers into the ComplexVec types.
                    for (buffer, packed_rbuf) in buffers.iter_mut().zip(packed_rbufs) {
                        buffer.manual_unpack(&packed_rbuf);
                    }
                }
                BenchmarkKind::Custom => {
                    let mut reqs = vec![];
                    for rbuf in buffers {
                        reqs.push(ctx.irecv(rbuf, 0, 0).expect("failed to receive buffer from rank 0"));
                    }
                    let _ = ctx.waitall(&reqs);
                }
            }

            let ack_buf = ComplexVec(vec![vec![2]; 1]);
            let ack_req = ctx.isend(&ack_buf, 0, 0).expect("failed to send ack to rank 0");
            let _ = ctx.waitall(&[ack_req]);
        }
    }
}

impl<R, C> BandwidthBenchmark for Benchmark<R, C>
where
    C: Communicator<Request = R>,
{
    fn init(&mut self, window_size: usize, size: usize) {
        match self.buffers {
            BenchmarkDatatypeBuffer::DoubleVec(ref mut buffers) => {
                let count = size / std::mem::size_of::<i32>();
                let subvector_count = self.subvector_size / std::mem::size_of::<i32>();

                if let Some(buffers) = buffers.as_mut() {
                    // Assume the window size doesn't change between iterations.
                    assert_eq!(buffers.len(), window_size);
    
                    for (i, buf) in buffers.iter_mut().enumerate() {
                        buf.update(count, subvector_count);
                    }
                } else {
                    let new_buffers = (0..window_size)
                        .map(|i| ComplexVec::new(count, subvector_count))
                        .collect();
                    let _ = buffers.insert(new_buffers);
                }
            }
            BenchmarkDatatypeBuffer::StructVec(ref mut buffers) => {
                assert_eq!(size % STRUCT_VEC_PACKED_SIZE_TOTAL, 0);
                assert!(size >= STRUCT_VEC_PACKED_SIZE_TOTAL);
                let count = size / STRUCT_VEC_PACKED_SIZE_TOTAL;

                if let Some(buffers) = buffers.as_mut() {
                    assert_eq!(buffers.len(), window_size);
                    for (i, buf) in buffers.iter_mut().enumerate() {
                        buf.update(count);
                    }
                } else {
                    let new_buffers = (0..window_size)
                        .map(|i| StructVecArray::new(count))
                        .collect();
                    let _ = buffers.insert(new_buffers);
                }
            }
        }
    }

    fn body(&mut self) {
        match self.buffers {
            BenchmarkDatatypeBuffer::DoubleVec(ref mut buffers) => {
                let buffers = buffers.as_mut().expect("missing buffers");
                bandwidth(self.rank, &self.ctx, self.kind, buffers);
            }
            BenchmarkDatatypeBuffer::StructVec(ref mut buffers) => {
                let buffers = buffers.as_mut().expect("missing buffers");
                bandwidth(self.rank, &self.ctx, self.kind, buffers);
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

    let buffers = match args.datatype {
        BenchmarkDatatype::DoubleVec => BenchmarkDatatypeBuffer::DoubleVec(None),
        BenchmarkDatatype::StructVec => BenchmarkDatatypeBuffer::StructVec(None),
    };
    let benchmark = Benchmark {
        kind: args.kind,
        ctx,
        rank,
        subvector_size: args.subvector_size,
        buffers,
    };
    mpicd_rust_benchmarks::bw(opts, benchmark, rank);
}
