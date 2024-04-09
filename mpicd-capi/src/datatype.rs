//! Datatype management code.
use std::ffi::{c_int, c_void};
use crate::c;

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    _packfn: c::PackFn,
    _unpackfn: c::UnpackFn,
    _elem_size: c::Count,
    _elem_extent: c::Count,
    _regfn: c::RegFn,
    _datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    // TODO
    0
}
