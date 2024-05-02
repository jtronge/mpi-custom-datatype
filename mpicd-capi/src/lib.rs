//! Experimental Rust API for custom datatype serialization.
use log::info;
use mpicd::communicator::Communicator;
use std::ffi::{c_char, c_int};
use std::sync::Once;

mod consts;
mod datatype;
mod p2p;
mod c;
mod ccontext;
use ccontext::CContext;

/// NOTE: This is not thread-safe but is required for now, unfortunately, since
///       the C interface relies on a global context. One way to solve this would be
///       to wrap internal types in an Arc<Mutex<..>>, but this would likely hurt
///       performance, since every call would have to lock this.
pub(crate) static mut CONTEXT: Option<(mpicd::Context, CContext)> = None;

/// Once object used for context initialization.
static CONTEXT_START: Once = Once::new();

/// Function passing a reference to the context from the outer scope.
///
/// SAFETY: Must be used only between calls of MPI_Init() and MPI_Finalize().
pub(crate) unsafe fn with_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut mpicd::Context, &mut CContext) -> R,
{
    let ctx = CONTEXT.as_mut().unwrap();
    f(&mut ctx.0, &mut ctx.1)
}

/// Initialize the MPI context.
#[no_mangle]
pub unsafe extern "C" fn MPI_Init(_argc: *mut c_int, _argv: *mut *mut *mut c_char) -> c::ReturnStatus {
    // Initialize logging.
    env_logger::init();

    info!("MPI_Init()");
    CONTEXT_START.call_once(|| {
        let _ = CONTEXT.insert((mpicd::init().expect("failed to initailize the MPI context"), CContext::new()));
    });
    consts::SUCCESS
}

/// Finalize everything.
#[no_mangle]
pub unsafe extern "C" fn MPI_Finalize() -> c::ReturnStatus {
    info!("MPI_Finalize()");
    let _ = CONTEXT.take();
    consts::SUCCESS
}

/// Get the size for this communicator.
#[no_mangle]
pub unsafe extern "C" fn MPI_Comm_size(comm: c::Comm, size: *mut c_int) -> c::ReturnStatus {
    info!("MPI_Comm_size()");

    // Assume MPI_COMM_WORLD.
    assert_eq!(comm, consts::COMM_WORLD);

    if let Some((ctx, _)) = CONTEXT.as_ref() {
        *size = ctx.size();
        consts::SUCCESS
    } else {
        consts::ERR_INTERNAL
    }
}

/// Get the size for this rank.
#[no_mangle]
pub unsafe extern "C" fn MPI_Comm_rank(comm: c::Comm, rank: *mut c_int) -> c::ReturnStatus {
    info!("MPI_Comm_rank()");

    // Assume MPI_COMM_WORLD.
    assert_eq!(comm, consts::COMM_WORLD);

    if let Some((ctx, _)) = CONTEXT.as_ref() {
        *rank = ctx.rank();
        consts::SUCCESS
    } else {
        consts::ERR_INTERNAL
    }
}

/// Get a system time.
#[no_mangle]
pub unsafe extern "C" fn MPI_Wtime() -> f64 {
    let time = std::time::SystemTime::now();
    time
        .duration_since(std::time::UNIX_EPOCH)
        .expect("failed duration from unix epoch to system time")
        .as_secs_f64()
}

/// Block all processes for this communicator until all have reached the given point.
#[no_mangle]
pub unsafe extern "C" fn MPI_Barrier(comm: c::Comm) -> c::ReturnStatus {
    assert_eq!(comm, consts::COMM_WORLD);
    if let Some((ctx, _)) = CONTEXT.as_ref() {
        ctx.barrier();
        consts::SUCCESS
    } else {
        consts::ERR_INTERNAL
    }
}
