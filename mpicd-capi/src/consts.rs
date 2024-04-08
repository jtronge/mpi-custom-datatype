//! MPI constant values.
//!
//! IMPORTANT: These must mirror those in `mpicd-capi/include/mpi.h`.
use std::ffi::c_int;

pub const SUCCESS: c_int = 0;
pub const ERR_INTERNAL: c_int = 1;
