use clap::Parser;
use mpicd_rust_benchmarks::{
    RsmpiArgs, RsmpiDatatype, RsmpiLatencyBenchmarkBuffer, LatencyBenchmark,
    LatencyOptions, StructVecArray, StructSimpleArray, StructSimpleNoGapArray,
};
use mpi::traits::*;
use mpi::topology::Process;

struct Benchmark<C: Communicator> {
    comm: C,
    rank: i32,
    buffers: RsmpiLatencyBenchmarkBuffer,
}

fn latency<C, S, R>(rank: i32, proc: Process<C>, sbuf: &S, rbuf: &mut R)
where
    C: Communicator,
    S: Buffer,
    R: BufferMut,
{
    if rank == 0 {
        proc.send(sbuf);
        let _ = proc.receive_into(rbuf);
    } else {
        let _ = proc.receive_into(rbuf);
        proc.send(sbuf);
    }
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        match self.buffers {
            RsmpiLatencyBenchmarkBuffer::Bytes(ref mut buffers) => {
                let _ = buffers.insert((vec![0; size], vec![0; size]));
            }
            RsmpiLatencyBenchmarkBuffer::StructVec(ref mut buffers) => {
                let _ = buffers.insert((StructVecArray::new(size).0, StructVecArray::new(size).0));
            }
            RsmpiLatencyBenchmarkBuffer::StructSimple(ref mut buffers) => {
                let _ = buffers.insert((StructSimpleArray::new(size).0, StructSimpleArray::new(size).0));
            }
            RsmpiLatencyBenchmarkBuffer::StructSimpleNoGap(ref mut buffers) => {
                let _ = buffers.insert((StructSimpleNoGapArray::new(size).0, StructSimpleNoGapArray::new(size).0));
            }
        }
    }

    fn body(&mut self) {
        let next_rank = (self.rank + 1) % 2;
        let proc = self.comm.process_at_rank(next_rank);
        match self.buffers {
            RsmpiLatencyBenchmarkBuffer::Bytes(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing latency buffers");
                latency(self.rank, proc, sbuf, rbuf);
            }
            RsmpiLatencyBenchmarkBuffer::StructVec(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing latency buffers");
                latency(self.rank, proc, sbuf, rbuf);
            }
            RsmpiLatencyBenchmarkBuffer::StructSimple(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing latency buffers");
                latency(self.rank, proc, sbuf, rbuf);
            }
            RsmpiLatencyBenchmarkBuffer::StructSimpleNoGap(ref mut buffers) => {
                let (sbuf, rbuf) = buffers.as_mut().expect("missing latency buffers");
                latency(self.rank, proc, sbuf, rbuf);
            }
        }
    }
}

fn main() {
    let args = RsmpiArgs::parse();
    let opts: LatencyOptions = mpicd_rust_benchmarks::load_options(args.options_path);
    let universe = mpi::initialize().expect("failed to initialize rsmpi");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    assert_eq!(size, 2);

    let buffers = match args.datatype {
        RsmpiDatatype::Bytes => RsmpiLatencyBenchmarkBuffer::Bytes(None),
        RsmpiDatatype::StructVec => RsmpiLatencyBenchmarkBuffer::StructVec(None),
        RsmpiDatatype::StructSimple => RsmpiLatencyBenchmarkBuffer::StructSimple(None),
        RsmpiDatatype::StructSimpleNoGap => RsmpiLatencyBenchmarkBuffer::StructSimpleNoGap(None),
    };
    let benchmark = Benchmark {
        comm: world,
        rank,
        buffers,
    };
    mpicd_rust_benchmarks::latency(opts, benchmark, rank);
}
