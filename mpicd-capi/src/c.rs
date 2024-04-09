//! C interface types and definitions.
use std::ffi::{c_int, c_void};

/// Default return value type.
pub type ReturnStatus = c_int;

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = c_int;

pub type PackFn = extern "C" fn(Count, *const c_void, Count, *mut c_void, *mut Count, *mut *mut c_void) -> c_int;

pub type UnpackFn = extern "C" fn(Count, *const c_void, Count, *mut c_void, *mut *mut c_void) -> c_int;

pub type QueryFn = extern "C" fn(buf: *const c_void, size: usize, packed_size: *mut usize);

pub type RegFn =
    extern "C" fn(*const c_void, Count, *mut Count, *mut *mut c_void, *mut *mut c_void) -> c_int;

pub type Request = c_int;

pub type Comm = c_int;

/// MPI_Status struct.
#[repr(C)]
pub struct Status {
    source: c_int,
    tag: c_int,
    error: c_int,
}
