//! Experimental Rust API for custom datatype serialization.
use std::ffi::{c_char, c_int, c_void};
use std::sync::Once;
use mpicd::communicator::Communicator;
use log::info;

mod consts;
mod datatype;
mod p2p;

pub type Request = c_int;
pub type Comm = c_int;

/// MPI_Status struct.
#[repr(C)]
pub struct Status {
    source: c_int,
    tag: c_int,
    error: c_int,
}

/// NOTE: This is not thread-safe but is required for now, unfortunately, since
///       the C interface relies on a global context. One way to solve this would be
///       to wrap internal types in an Arc<Mutex<..>>, but this would likely hurt
///       performance, since every call would have to lock this.
pub(crate) static mut CONTEXT: Option<mpicd::Context> = None;
static CONTEXT_START: Once = Once::new();

/// Function passing a reference to the context from the outer scope.
///
/// SAFETY: Must be used only between calls of MPI_Init() and MPI_Finalize().
pub(crate) unsafe fn with_context<F, R>(f: F) -> R
where
    F: FnOnce(&mpicd::Context) -> R,
{
    f(CONTEXT.as_ref().unwrap())
}

/// Initialize the MPI context.
#[no_mangle]
pub unsafe extern "C" fn MPI_Init(_argc: *mut c_int, _argv: *mut *mut *mut c_char) -> c_int {
    // Initialize logging.
    env_logger::init();

    info!("MPI_Init()");
    CONTEXT_START.call_once(|| {
        let _ = CONTEXT.insert(mpicd::init().expect("failed to initailize the MPI context"));
    });
    consts::SUCCESS
}

/// Finalize everything.
#[no_mangle]
pub unsafe extern "C" fn MPI_Finalize() -> c_int {
    info!("MPI_Finalize()");
    let _ = CONTEXT.take();
    consts::SUCCESS
}

/// Get the size for this communicator.
///
/// NOTE: For now, all communicators are assumed to be MPI_COMM_WORLD.
#[no_mangle]
pub unsafe extern "C" fn MPI_Comm_size(_comm: Comm, size: *mut c_int) -> c_int {
    info!("MPI_Comm_size()");
    if let Some(ctx) = CONTEXT.as_ref() {
        *size = ctx.size();
        consts::SUCCESS
    } else {
        consts::ERR_INTERNAL
    }
}

/// Get the size for this rank.
///
/// NOTE: For now, all communicators are assumed to be MPI_COMM_WORLD.
#[no_mangle]
pub unsafe extern "C" fn MPI_Comm_rank(_comm: Comm, rank: *mut c_int) -> c_int {
    info!("MPI_Comm_rank()");
    if let Some(ctx) = CONTEXT.as_ref() {
        *rank = ctx.rank();
        consts::SUCCESS
    } else {
        consts::ERR_INTERNAL
    }
}
