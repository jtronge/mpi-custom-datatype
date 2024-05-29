use clap::{Parser, ValueEnum};
use serde::de::DeserializeOwned;
use std::path::Path;

mod latency;
pub use latency::{latency, LatencyOptions, LatencyBenchmark};
mod bw;
pub use bw::{bw, BandwidthOptions, BandwidthBenchmark};
mod datatype;
pub use datatype::{
    ManualPack, ComplexVec, StructVecArray, BenchmarkDatatypeBuffer, RsmpiDatatypeBuffer,
    STRUCT_VEC_DATA_COUNT, STRUCT_VEC_PACKED_SIZE_TOTAL,
};
mod random;

/// Generic benchmark args.
#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Type of benchmark to run.
    #[arg(value_enum, short, long)]
    pub kind: BenchmarkKind,

    /// Datatype to use.
    #[arg(short, long)]
    pub datatype: BenchmarkDatatype,

    /// Path for benchmark options file.
    #[arg(short, long)]
    pub options_path: String,

    /// Number of sub-vectors to use.
    #[arg(short, long)]
    pub subvector_size: usize,
}

/// RSMPI specific args.
#[derive(Parser)]
pub struct RsmpiArgs {
    /// Path for benchmark options file.
    #[arg(short, long)]
    pub options_path: String,

    /// Datatype to use.
    #[arg(short, long)]
    pub datatype: RsmpiDatatype,
}

/// Kind of benchmark to run.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum BenchmarkKind {
    /// Manual packing benchmark.
    Packed,

    /// Custom datatype benchmark.
    Custom,
}

/// Datatype to use for the benchmark.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum BenchmarkDatatype {
    /// Use the double vec type (a.k.a. ComplexVec).
    DoubleVec,

    /// Use the struct and vec type.
    StructVec,
}

/// Datatype to use for the rsmpi benchmarks.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum RsmpiDatatype {
    /// Plain bytes datatype.
    Bytes,

    /// Struct vec datatype with Equivalence implementation.
    StructVec,
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
