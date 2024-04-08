//! Datatype management code.
use std::ffi::{c_int, c_void};

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = *mut c_void;

pub type PackFn = extern "C" fn(Count, *const c_void, *mut c_void, *mut Count, *mut *mut c_void);

pub type UnpackFn = extern "C" fn(Count, *const c_void, *mut c_void, *mut Count, *mut *mut c_void);

pub type RegFn =
    extern "C" fn(*const c_void, Count, *mut Count, *mut *mut c_void, *mut *mut c_void);

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    _packfn: PackFn,
    _unpackfn: UnpackFn,
    _elem_size: Count,
    _elem_extent: Count,
    _regfn: RegFn,
    _datatype: *mut Datatype,
) -> c_int {
    // TODO
    0
}
