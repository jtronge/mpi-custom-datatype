//! C interface types and definitions.
use std::ffi::{c_int, c_void};

/// Default return value type.
pub type ReturnStatus = c_int;

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = c_int;

/// Function for creating the pack state from a context and buffer.
pub type PackStateFn = Option<
    extern "C" fn(
        context: *mut c_void,
        src: *const c_void,
        src_count: Count,
        state: *mut *mut c_void,
    ) -> c_int
>;

/// Function for creating the unpack state from a context and buffer.
pub type UnpackStateFn = Option<
    extern "C" fn(
        context: *mut c_void,
        dst: *mut c_void,
        dst_count: Count,
        state: *mut *mut c_void,
    ) -> c_int
>;

/// Function for querying the total packed size of a buffer.
pub type QueryFn = Option<
    extern "C" fn(
        context: *mut c_void,
        buf: *const c_void,
        count: Count,
        packed_size: *mut Count,
    ) -> c_int
>;

/// Actual pack function.
pub type PackFn = Option<
    extern "C" fn(
        state: *mut c_void,
        offset: Count,
        dst: *mut c_void,
        dst_size: Count,
        used: *mut Count,
    ) -> c_int
>;

/// Actual unpack function.
pub type UnpackFn = Option<
    extern "C" fn(
        state: *mut c_void,
        offset: Count,
        src: *const c_void,
        src_size: Count,
    ) -> c_int
>;

/// Free the pack state.
pub type PackStateFreeFn = Option<extern "C" fn(state: *mut c_void) -> c_int>;

/// Free the unpack state.
pub type UnpackStateFreeFn = Option<extern "C" fn(state: *mut c_void) -> c_int>;

pub type Request = c_int;

pub type Comm = c_int;

/// MPI_Status struct.
#[repr(C)]
pub struct Status {
    source: c_int,
    tag: c_int,
    error: c_int,
}
