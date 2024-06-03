//! mpicd library code and entry points.
use log::{error, info};
use mpicd_ucx_sys::{
    rust_ucp_init, ucp_address_t, ucp_cleanup, ucp_context_h, ucp_ep_close_nb,
    ucp_ep_create, ucp_ep_h, ucp_ep_params_t, ucp_params_t, ucp_tag_t,
    ucp_worker_create, ucp_worker_destroy, ucp_worker_get_address, ucp_worker_h,
    ucp_worker_params_t, ucp_worker_release_address, ucs_status_string,
    ucs_status_t, UCP_EP_CLOSE_MODE_FORCE, UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE,
    UCP_EP_PARAM_FIELD_REMOTE_ADDRESS, UCP_ERR_HANDLING_MODE_PEER,
    UCP_FEATURE_STREAM, UCP_FEATURE_TAG, UCP_PARAM_FIELD_FEATURES,
    UCP_WORKER_PARAM_FIELD_THREAD_MODE, UCS_OK, UCS_THREAD_MODE_SINGLE,
};
use std::cell::RefCell;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::rc::Rc;

pub type Tag = ucp_tag_t;

pub mod communicator;
mod context;
pub use context::Context;
mod util;
use util::wait_loop;
pub mod datatype;
mod pmi;
use pmi::PMI;
mod request;
mod message;
use message::Message;

/// Status value for requests and messages.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Request is in progress.
    InProgress,

    /// Request has completed.
    Complete,

    /// Error occurred.
    Error(String),
}

#[derive(Debug, Copy, Clone)]
pub enum Error {
    /// Initialization failure.
    InitFailure,

    /// An error was returned by the UCX library.
    UCXError(ucs_status_t),

    /// Worker creation failed.
    WorkerCreateFailed(ucs_status_t),

    /// Address exchange failed.
    WorkerAddressFailure(ucs_status_t),

    /// Waiting for a worker failed.
    WorkerWait(ucs_status_t),

    /// A request failed.
    FailedRequest(ucs_status_t),

    /// Timeout occured while waiting on a request.
    RequestTimeout,

    /// Internal error occurred.
    InternalError,
}

/// UCP-specific data.
pub(crate) struct System {
    /// PMI Context.
    pub _pmi: PMI,

    /// UCP context.
    pub context: ucp_context_h,

    /// UCP worker.
    pub worker: ucp_worker_h,

    /// UCP endpoints.
    pub endpoints: Vec<ucp_ep_h>,

    /// Number of processes.
    pub size: usize,

    /// Rank of this process.
    pub rank: usize,
}

impl Drop for System {
    fn drop(&mut self) {
        unsafe {
            // Destroy endpoints.
            for ep in &self.endpoints {
                // For some reason UCP_EP_CLOSE_MODE_FLUSH is causing an
                // infinite loop with two nodes.
                // let req = ucp_ep_close_nb(endpoint, UCP_EP_CLOSE_MODE_FLUSH);
                let req = ucp_ep_close_nb(ep.clone(), UCP_EP_CLOSE_MODE_FORCE);
                wait_loop(self.worker, req, || false).unwrap();
            }
            ucp_worker_destroy(self.worker);
            ucp_cleanup(self.context);
        }
    }
}

/// Handle containing the internal UCP context data and other code.
pub(crate) struct Handle {
    /// System data structures.
    pub system: System,

    /// Current messages.
    pub messages: Vec<Option<Box<dyn Message>>>,

    /// Index of free messages.
    pub free_messages: Vec<usize>,
}

impl Handle {
    /// Add a new request pointer.
    pub(crate) fn add_message(&mut self, message: impl Message + 'static) -> usize {
        let message = Box::new(message);
        if let Some(i) = self.free_messages.pop() {
            assert!(self.messages[i].is_none());
            let _ = self.messages[i].insert(message);
            i
        } else {
            let i = self.messages.len();
            self.messages.push(Some(message));
            i
        }
    }

    /// Make progress for the specific message.
    pub(crate) unsafe fn message_progress(&mut self, msg_id: usize) -> Status {
        let msg = self.messages[msg_id].as_mut().expect("request is missing");
        msg.progress(&mut self.system)
    }

    /// Remove a completed message.
    pub(crate) fn remove_message(&mut self, msg_id: usize) {
        let _ = self.messages[msg_id].take();
        self.free_messages.push(msg_id);
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // Free requests (this must be done before freeing endpoints, etc.).
        for msg in &mut self.messages {
            if let Some(msg) = msg.take() {
                drop(msg);
            }
        }
        // System data should be dropped here.
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Initialize the safe mpi context.
pub fn init() -> Result<Context> {
    unsafe {
        // Attempt to initialize the PMI.
        let pmi = PMI::init();

        let mut context = MaybeUninit::<ucp_context_h>::uninit();
        let params = ucp_params_t {
            field_mask: UCP_PARAM_FIELD_FEATURES.into(),
            features: (UCP_FEATURE_TAG | UCP_FEATURE_STREAM).into(),
            ..Default::default()
        };
        let status = rust_ucp_init(&params, std::ptr::null(), context.as_mut_ptr());
        if status != UCS_OK {
            error!("Failed to create context: {}", status_to_string(status));
            Err(Error::InitFailure)
        } else {
            let context = context.assume_init();
            let worker = create_worker(context)?;
            let worker_addr = get_worker_address(worker)?;

            let rank = pmi.rank();
            let size = pmi.size();

            // Exchange worker addresses.
            pmi.put("UCP_WORKER_ADDR", worker_addr.clone());
            info!(
                "(rank = {}) Putting address: {:?}",
                rank,
                &worker_addr[..10]
            );
            pmi.fence();

            let mut endpoints = vec![];
            for ep_rank in 0..size {
                if ep_rank == rank {
                    endpoints.push(create_endpoint(worker, &worker_addr));
                } else {
                    let addr: Vec<u8> = pmi.get(ep_rank as u32, "UCP_WORKER_ADDR");
                    info!(
                        "(rank = {}) Got address for other proc: {:?}",
                        rank,
                        &addr[..10]
                    );
                    endpoints.push(create_endpoint(worker, &addr));
                }
            }

            Ok(Context::new(Rc::new(RefCell::new(Handle {
                system: System {
                    _pmi: pmi,
                    context,
                    worker,
                    endpoints,
                    size: size as usize,
                    rank: rank as usize,
                },
                messages: vec![],
                free_messages: vec![],
            }))))
        }
    }
}

/// Create the worker.
unsafe fn create_worker(context: ucp_context_h) -> Result<ucp_worker_h> {
    // First create the worker
    let mut worker = MaybeUninit::<ucp_worker_h>::uninit();
    let params = ucp_worker_params_t {
        field_mask: UCP_WORKER_PARAM_FIELD_THREAD_MODE.into(),
        // One thread for now
        thread_mode: UCS_THREAD_MODE_SINGLE,
        ..Default::default()
    };
    let status = ucp_worker_create(context, &params, worker.as_mut_ptr());
    if status != UCS_OK {
        Err(Error::WorkerCreateFailed(status))
    } else {
        Ok(worker.assume_init())
    }
}

/// Return the address for the worker.
unsafe fn get_worker_address(worker: ucp_worker_h) -> Result<Vec<u8>> {
    // Get the address of the worker.
    let mut addr = MaybeUninit::<*mut ucp_address_t>::uninit();
    let mut len = MaybeUninit::<usize>::uninit();
    let status = ucp_worker_get_address(worker, addr.as_mut_ptr(), len.as_mut_ptr());
    if status != UCS_OK {
        return Err(Error::WorkerAddressFailure(status));
    }
    let addr = addr.assume_init();
    let len = len.assume_init();
    let mut buffer = Vec::with_capacity(len);
    std::ptr::copy(addr as *const u8, buffer.as_mut_ptr(), len);
    buffer.set_len(len);
    ucp_worker_release_address(worker, addr);
    Ok(buffer)
}

/// Create an endpoint for the worker and remote address.
unsafe fn create_endpoint(worker: ucp_worker_h, remote_addr: &[u8]) -> ucp_ep_h {
    let mut endpoint = MaybeUninit::<ucp_ep_h>::uninit();
    let params = ucp_ep_params_t {
        field_mask: (UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE)
            .into(),
        err_mode: UCP_ERR_HANDLING_MODE_PEER,
        address: remote_addr.as_ptr() as *const _,
        ..Default::default()
    };
    let status = ucp_ep_create(worker, &params, endpoint.as_mut_ptr());
    if status != UCS_OK {
        panic!(
            "Failed to create endpoint for worker: {}",
            status_to_string(status)
        );
    }
    endpoint.assume_init()
}

/// Convert the ucs_status_t to a Rust string.
pub(crate) fn status_to_string(status: ucs_status_t) -> String {
    unsafe {
        CStr::from_ptr(ucs_status_string(status))
            .to_str()
            .expect("Failed to convert status string")
            .to_string()
    }
}
