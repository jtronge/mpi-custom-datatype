//! Latency benchmark code
use serde::Deserialize;
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct LatencyOptions {
    pub iterations: usize,
    pub skip: usize,
    pub warmup_validation: usize,
    pub min_size: usize,
    pub max_size: usize,
}

pub trait LatencyBenchmark {
    /// Initialization code (not timed).
    fn init(&mut self, size: usize);

    /// Code of the benchmark being timed.
    fn body(&mut self);
}

/// Generic latency benchmark function. Returns a vec of pairs of the form
/// (size, microseconds).
///
/// Based on the OSU microbenchmarks version for MPI.
pub fn latency(
    opts: LatencyOptions,
    mut benchmark: impl LatencyBenchmark,
    rank: i32,
) {
    let mut size = opts.min_size;

    if rank == 0 {
        println!("# size latency");
    }
    while size <= opts.max_size {
        let mut total_time = 0.0;
        // Prepare to run the benchmark code.
        benchmark.init(size);
        for i in 0..opts.iterations + opts.skip {
            for j in 0..=opts.warmup_validation {
                let start = Instant::now();
                // Body of code being benchmarked.
                benchmark.body();
                if i >= opts.skip && j == opts.warmup_validation {
                    total_time += Instant::now().duration_since(start).as_secs_f32();
                }
            }
        }
        let latency = (total_time * 1.0e6) / (2.0 * opts.iterations as f32);
        if rank == 0 {
            println!("{} {}", size, latency);
        }
        size *= 2;
    }
}
