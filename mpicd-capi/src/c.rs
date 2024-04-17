//! C interface types and definitions.
use std::ffi::{c_int, c_void};

/// Default return value type.
pub type ReturnStatus = c_int;

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = c_int;

/// Function pointer for packing a custom datatype buffer.
pub type PackFn = Option<
    extern "C" fn(
        src_size: Count,
        src: *const c_void,
        dst_size: Count,
        dst: *mut c_void,
        resume: *mut *mut c_void,
    ) -> c_int
>;

/// Function pointer for unpacking a custom datatype buffer.
pub type UnpackFn = Option<
    extern "C" fn(
        src_size: Count,
        src: *const c_void,
        dst_size: Count,
        dst: *mut c_void,
        resume: *mut *mut c_void,
    ) -> c_int
>;

/// Function pointer for querying the size of a packed representation buffer.
pub type QueryFn = Option<
    extern "C" fn(buf: *const c_void, size: usize, packed_size: *mut usize) -> c_int
>;

/// Function pointer for getting iovec-like memory regions for a type.
pub type RegFn = Option<
    extern "C" fn(
        src: *const c_void,
        max_regions: Count,
        region_lengths: *mut Count,
        region_bases: *mut *mut c_void,
        resume: *mut *mut c_void,
    ) -> c_int
>;

pub type Request = c_int;

pub type Comm = c_int;

/// MPI_Status struct.
#[repr(C)]
pub struct Status {
    source: c_int,
    tag: c_int,
    error: c_int,
}
