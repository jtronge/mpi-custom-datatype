use clap::Parser;
use mpi::traits::*;
use mpicd_rust_benchmarks::{
    RsmpiArgs, RsmpiDatatype, RsmpiDatatypeBuffer, BandwidthOptions, BandwidthBenchmark,
    StructVecArray, StructSimpleArray,
};

fn bandwidth<C, B>(rank: i32, comm: &C, buffers: &mut Vec<B>)
where
    C: Communicator,
    B: Buffer + BufferMut,
{
    let window_size = buffers.len();
    let proc = comm.process_at_rank((rank + 1) % 2);
    mpi::request::multiple_scope(window_size, |scope, coll| {
        if rank == 0 {
            for i in 0..window_size {
                coll.add(proc.immediate_send(scope, &buffers[i]));
            }
        } else {
            let mut tmp = &mut buffers[..];
            for _ in 0..window_size {
                let (first, second) = tmp.split_at_mut(1);
                tmp = second;
                coll.add(proc.immediate_receive_into(scope, &mut first[0]));
            }
        }
        let mut stats = vec![];
        coll.wait_all(&mut stats);
    });

    if rank == 0 {
        let (_, _): (Vec<i32>, _) = proc.receive_vec();
    } else {
        proc.send(&[0i32]);
    }
}

struct RsmpiBenchmark<C: Communicator> {
    comm: C,
    rank: i32,
    buffers: RsmpiDatatypeBuffer,
}

impl<C: Communicator> BandwidthBenchmark for RsmpiBenchmark<C> {
    fn init(&mut self, window_size: usize, size: usize) {
        match self.buffers {
            RsmpiDatatypeBuffer::Bytes(ref mut buffers) => {
                let buf = (0..window_size).map(|_| vec![0u8; size]).collect();
                let _ = buffers.insert(buf);
            }
            RsmpiDatatypeBuffer::StructVec(ref mut buffers) => {
                let buf = (0..window_size).map(|_| StructVecArray::new(size).0).collect();
                let _ = buffers.insert(buf);
            }
            RsmpiDatatypeBuffer::StructSimple(ref mut buffers) => {
                let buf = (0..window_size).map(|_| StructSimpleArray::new(size).0).collect();
                let _ = buffers.insert(buf);
            }
        }
    }

    fn body(&mut self) {
        match self.buffers {
            RsmpiDatatypeBuffer::Bytes(ref mut buffers) => {
                let buffers = buffers.as_mut().expect("missing bytes buffer");
                bandwidth(self.rank, &self.comm, buffers);
            }
            RsmpiDatatypeBuffer::StructVec(ref mut buffers) => {
                let buffers = buffers.as_mut().expect("missing struct-vec buffer");
                bandwidth(self.rank, &self.comm, buffers);
            }
            RsmpiDatatypeBuffer::StructSimple(ref mut buffers) => {
                let buffers = buffers.as_mut().expect("missing struct-vec buffer");
                bandwidth(self.rank, &self.comm, buffers);
            }
        }
    }
}

fn main() {
    let args = RsmpiArgs::parse();
    let opts: BandwidthOptions = mpicd_rust_benchmarks::load_options(&args.options_path);
    let universe = mpi::initialize().expect("failed to initialize rsmpi");
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    assert_eq!(size, 2);

    let buffers = match args.datatype {
        RsmpiDatatype::Bytes => RsmpiDatatypeBuffer::Bytes(None),
        RsmpiDatatype::StructVec => RsmpiDatatypeBuffer::StructVec(None),
        RsmpiDatatype::StructSimple => RsmpiDatatypeBuffer::StructSimple(None),
    };
    let benchmark = RsmpiBenchmark {
        comm: world,
        rank,
        buffers,
    };
    mpicd_rust_benchmarks::bw(opts, benchmark, rank);
}
