//! C interface types and definitions.
use std::ffi::{c_int, c_void};

/// Default return value type.
pub type ReturnStatus = c_int;

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = c_int;

/// Function for creating the pack state from a context and buffer.
pub type StateFn = Option<
    unsafe extern "C" fn(
        context: *mut c_void,
        buf: *const c_void,
        count: Count,
        state: *mut *mut c_void,
    ) -> c_int
>;

/// Free the pack state.
pub type StateFreeFn = Option<unsafe extern "C" fn(state: *mut c_void) -> c_int>;

/// Function for querying the total packed size of a buffer.
pub type QueryFn = Option<
    unsafe extern "C" fn(
        state: *mut c_void,
        buf: *const c_void,
        count: Count,
        packed_size: *mut Count,
    ) -> c_int
>;

/// Actual pack function.
pub type PackFn = Option<
    unsafe extern "C" fn(
        state: *mut c_void,
        buf: *const c_void,
        count: Count,
        offset: Count,
        dst: *mut c_void,
        dst_size: Count,
        used: *mut Count,
    ) -> c_int
>;

/// Actual unpack function.
pub type UnpackFn = Option<
    unsafe extern "C" fn(
        state: *mut c_void,
        buf: *mut c_void,
        count: Count,
        offset: Count,
        src: *const c_void,
        src_size: Count,
    ) -> c_int
>;

/// Get the number of memory regions.
pub type RegionCountFn = Option<
    unsafe extern "C" fn(
        state: *mut c_void,
        buf: *mut c_void,
        count: Count,
        region_count: *mut Count,
    ) -> c_int
>;

/// Receive handler for the iovec-like API.
///
/// NOTE: One problem with this functio and the iovec interface in general is
///       the requirement that the buffer always be mutable, even when it will not be
///       written to for send functions. As long as the buffer is only written to when
///       expected, then I don't think this is a problem, but it could be a source of
///       UB.
pub type RegionFn = Option<
    unsafe extern "C" fn(
        state: *mut c_void,
        buf: *mut c_void,
        count: Count,
        region_count: Count,
        reg_lens: *mut Count,
        reg_bases: *mut *mut c_void,
        types: *mut Datatype,
    ) -> c_int
>;

pub type Request = isize;

pub type Comm = c_int;

/// MPI_Status struct.
#[repr(C)]
pub struct Status {
    pub count: c_int,
    pub cancelled: c_int,
    pub source: c_int,
    pub tag: c_int,
    pub error: c_int,
}
