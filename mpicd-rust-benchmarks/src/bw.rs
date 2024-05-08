//! Bandwidth benchmark code
use serde::Deserialize;
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct BandwidthOptions {
    pub min_size: usize,
    pub max_size: usize,
    pub window_size: usize,
    pub iterations: usize,
    pub skip: usize,
    pub warmup_validation: usize,
}

pub trait BandwidthBenchmark {
    /// Initialize any data needed for the body of the benchmark.
    fn init(&mut self, window_size: usize, size: usize);

    /// Run the code to be benchmarked.
    fn body(&mut self);
}

/// Generic bandwidth benchmark function. Returns a vec of pairs of the form
/// (size, MB/s).
///
/// Based on the OSU microbenchmarks version for MPI.
pub fn bw(
    opts: BandwidthOptions,
    mut benchmark: impl BandwidthBenchmark,
) -> Vec<(usize, f32)> {
    let mut results = vec![];
    let mut size = opts.min_size;

    while size <= opts.max_size {
        let mut total_time = 0.0;
        // Prepare to run the benchmark.
        benchmark.init(opts.window_size, size);
        for i in 0..(opts.iterations + opts.skip) {
            for k in 0..=opts.warmup_validation {
                let start = Instant::now();
                benchmark.body();
                if i >= opts.skip && k == opts.warmup_validation {
                    // The osu version includes another factor that I'm not
                    // sure is necessary.
                    total_time += Instant::now().duration_since(start).as_secs_f32();
                }
            }
        }
        let bandwidth =
            (size as f32 / 1.0e6 * opts.iterations as f32 * opts.window_size as f32)
                / total_time;
        results.push((size, bandwidth));
        size *= 2;
    }

    results
}
