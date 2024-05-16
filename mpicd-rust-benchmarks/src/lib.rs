use clap::{Parser, ValueEnum};
use serde::de::DeserializeOwned;
use std::path::Path;

mod latency;
pub use latency::{latency, LatencyOptions, LatencyBenchmark};
mod bw;
pub use bw::{bw, BandwidthOptions, BandwidthBenchmark};
mod datatype;
pub use datatype::{ComplexVec, IovecComplexVec, IovecComplexVecMut};
mod random;

/// Generic benchmark args.
#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Type of benchmark to run.
    #[arg(value_enum, short, long)]
    pub kind: BenchmarkKind,

    /// Path for benchmark options file.
    #[arg(short, long)]
    pub options_path: String,

    /// Should this test use a single vector?
    #[arg(short, long)]
    pub single_vec: bool,
}

/// RSMPI specific args.
#[derive(Parser)]
pub struct RsmpiArgs {
    /// Path for benchmark options file.
    #[arg(short, long)]
    pub options_path: String,
}

/// Kind of benchmark to run.
#[derive(Clone, Debug, ValueEnum)]
pub enum BenchmarkKind {
    /// Manual packing benchmark.
    Packed,

    /// Custom datatype benchmark.
    Custom,

    /// Iovec benchmark.
    Iovec,
}

/// Load benchmark options from a file path.
pub fn load_options<P, T>(path: P) -> T
where
    P: AsRef<Path>,
    T: DeserializeOwned,
{
    let fp = std::fs::File::open(path).expect("failed to load option file");
    serde_yaml::from_reader(fp).expect("failed to deserialize option file")
}
