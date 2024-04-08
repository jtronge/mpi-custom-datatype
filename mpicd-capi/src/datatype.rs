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
    packfn: PackFn,
    unpackfn: UnpackFn,
    elem_size: Count,
    elem_extent: Count,
    regfn: RegFn,
    datatype: *mut Datatype,
) -> c_int {
    // TODO
    0
}

/*
pub mod dynamic_type {
    //! Code used for creating dynamic custom MPI_Datatype's.
    use super::*;

    pub type PackFn = extern "C" fn(Count, *const c_void, *mut c_void, Count, *mut Count, *mut *mut c_void);
    pub type UnpackFn = extern "C" fn(Count, *const c_void, *mut c_void, Count, *mut *mut c_void);
    pub type QueryFn = extern "C" fn(*const c_void, *mut Count);
    pub type NumRegFn = extern "C" fn(*const c_void, *mut Count);
    pub type RegFn = extern "C" fn(Count, *const c_void, *mut *mut Count, *mut *mut c_void, *mut *mut c_void);

    /// Create a dynamic custom MPI_Datatype.
    #[no_mangle]
    pub unsafe extern "C" fn MPI_Type_create_dynamic(
        packfn: PackFn,
        unpackfn: UnpackFn,
        sizefn: QueryFn,
        rcfn: NumRegFn,
        regfn: RegFn,
        elem_pack_size: Count,
        elem_extent: Count,
        datatype: *mut Datatype,
    ) -> c_int {
        0
    }
}
*/
