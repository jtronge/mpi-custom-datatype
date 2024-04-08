//! Experimental Rust API for custom datatype serialization.
use std::ffi::{c_void, c_int, c_char};
use std::sync::Once;

/// Count type corresponding to MPI_Count.
pub type Count = usize;

/// Type corresponding to MPI_Type.
pub type Datatype = *mut c_void;

pub type Comm = *mut c_void;

pub type Request = *mut c_void;

pub type PackFn = extern "C" fn(Count, *const c_void, *mut c_void, *mut Count, *mut *mut c_void);

pub type UnpackFn = extern "C" fn(Count, *const c_void, *mut c_void, *mut Count, *mut *mut c_void);

pub type RegFn = extern "C" fn(*const c_void, Count, *mut Count, *mut *mut c_void, *mut *mut c_void);

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

// NOTE: This is not thread-safe but is required for now, unfortunately, since
//       the C interface relies on a global context. One way to solve this would be
//       to wrap internal types in an Arc<Mutex<..>>, but this would likely hurt
//       performance, since every call would have to lock this.
static mut CONTEXT: Option<mpicd::Context> = None;
static CONTEXT_START: Once = Once::new();
//= Some((mpicd::init().expect("failed to initialize MPI context"))));

#[no_mangle]
pub unsafe extern "C" fn MPI_Init(_argc: *mut c_int, _argv: *mut *mut *mut c_char) -> c_int {
    CONTEXT_START.call_once(|| {
        unsafe {
            let _ = CONTEXT.insert(mpicd::init().expect("failed to initailize the MPI context"));
        }
    });
    println!("in rust code...");
    0
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Finalize() -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Isend(buf: *mut c_void, count: c_int, datatype: Datatype, dest: c_int, tag: c_int, comm: Comm, request: *mut Request) -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Irecv(buf: *const c_void, count: c_int, datatype: Datatype, source: c_int, tag: c_int, comm: Comm, request: *mut Request) -> c_int {
    0
}

#[repr(C)]
pub struct Status {
    source: c_int,
    tag: c_int,
    error: c_int,
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Waitall(count: c_int, array_of_requests: *mut Request, array_of_statuses: *mut Status) -> c_int {
    0
}
