use mpicd::communicator::Communicator;
use mpicd_rust_benchmarks::{ComplexVecType, latency, LatencyBenchmark, LatencyOptions};

struct Random {
    last: usize,
}

impl Random {
    fn new(seed: usize) -> Random {
        Random {
            last: seed,
        }
    }

    fn value(&mut self) -> usize {
        let v = (503 * self.last) % 54018521;
        self.last = v;
        v
    }
}

/// Generate a randomly partitioned complex vec type.
fn generate_complex_vec(total: usize, seed: usize) -> ComplexVecType {
    let mut rand = Random::new(seed);
    let mut count = 0;
    let mut data = vec![];
    while count < total {
        let len = rand.value() % total;
        let len = if (count + len) > total {
            total - count
        } else {
            len
        };
        let inner_data = (0..len).map(|i| (count + i) as i32).collect();
        data.push(inner_data);
        count += len;
    }
    ComplexVecType(data)
}

struct Benchmark<C: Communicator> {
    ctx: C,
    size: i32,
    rank: i32,
    sbuf: Option<ComplexVecType>,
    rbuf: Option<ComplexVecType>,
}

impl<C: Communicator> LatencyBenchmark for Benchmark<C> {
    fn init(&mut self, size: usize) {
        let _ = self.sbuf.insert(generate_complex_vec(size, 2333));
        let _ = self.rbuf.insert(ComplexVecType(vec![vec![0; size]]));
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

                // let buffer = self.buffer.as_mut().expect("missing buffer");
                // let req = self.ctx.irecv(buffer, 0, 0).expect("failed to receive buffer from rank 0");
                // let _ = self.ctx.waitall(&[req]);
                // println!("buf = {:?}", &self.buffer.as_ref().expect("missing buffer").0[0][..2]);
            }
        }
    }
}

fn main() {
    let ctx = mpicd::init().expect("failed to init mpirs");
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
/*
        rank.try_into().unwrap(),
        |size| generate_complex_vec(size, 2333),
        |data| {
            unsafe {
                let req = ctx.isend(&data, 1, 0).expect("failed to send buffer to rank 1");
                let _ = ctx.waitall(&[req]);
            }
        },
        |data| {
            unsafe {
                let req = ctx.irecv(&mut data, 0, 0).expect("failed to receive buffer from rank 0");
                let _ = ctx.waitall(&[req]);
                println!("buf = {:?}", &buf.0[0][..10]);
                for i in 0..8192 {
                    assert_eq!(buf.0[0][i], i as i32);
                }
            }
        },
    );
*/
}
