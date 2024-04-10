//! MPI constant values.
//!
//! IMPORTANT: These must mirror those in `mpicd-capi/include/mpi.h`.
use std::ffi::c_int;
use crate::c;

pub const SUCCESS: c::ReturnStatus = 0;

pub const ERR_INTERNAL: c::ReturnStatus = 1;

pub const COMM_WORLD: c::Comm = 1;

pub const BYTE: c::Datatype = 1;

pub const MAX_PREDEFINED: c::Datatype = 1;
