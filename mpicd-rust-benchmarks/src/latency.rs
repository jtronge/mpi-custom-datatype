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
    fn init(&mut self, size: usize);

    fn body(&mut self);
}

/// Generic latency benchmark function. Returns a vec of pairs of the form
/// (size, microseconds).
///
/// The `prepare` callback is used to prepare data for an iteration. The
/// `body0` and `body1` callbacks are called on rank 0 and 1 of the
/// communicator respectively.
///
/// Based on the OSU microbenchmarks version for MPI.
pub fn latency(
    opts: LatencyOptions,
    mut benchmark: impl LatencyBenchmark,
) -> Vec<(usize, f32)> {
    let mut results = vec![];
    let mut size = opts.min_size;
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
        results.push((size, latency));
        size *= 2;
    }
    results
}
