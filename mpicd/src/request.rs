//! Request object.
use std::ffi::c_void;
use mpicd_ucx_sys::{rust_ucs_ptr_is_err, rust_ucs_ptr_is_ptr, rust_ucs_ptr_status, ucp_tag_recv_info_t, ucp_request_free, ucs_status_t, UCS_OK, UCS_INPROGRESS};
use crate::{status_to_string, datatype::UCXBuffer};

/// Request status value.
#[derive(Clone, Debug, PartialEq)]
pub enum RequestStatus {
    /// Request is in progress.
    InProgress,

    /// Request has completed.
    Complete,

    /// Error occurred.
    Error(String),
}

/// Request struct.
pub(crate) struct Request {
    /// Request pointer.
    request: *mut c_void,

    /// TODO: Ideally this should be an AtomicBool
    data: Option<*mut RequestData>,
}

impl Request {
    /// Create a new request from a ucp pointer and request data.
    pub(crate) fn new(request: *mut c_void, data: Option<*mut RequestData>) -> Request {
        Request {
            request,
            data,
        }
    }

    /// Get the status of this request.
    pub(crate) unsafe fn status(&self) -> RequestStatus {
        if rust_ucs_ptr_is_ptr(self.request) == 0 {
            let status = rust_ucs_ptr_status(self.request);
            if status != UCS_OK {
                RequestStatus::Error(status_to_string(status))
            } else {
                RequestStatus::Complete
            }
        } else if rust_ucs_ptr_is_err(self.request) != 0 {
            RequestStatus::Error("Internal pointer failure".to_string())
        } else {
            let status = rust_ucs_ptr_status(self.request);
            if !(**self.data.as_ref().unwrap()).complete {
                if status == UCS_OK {
                    RequestStatus::Complete
                } else if status == UCS_INPROGRESS {
                    RequestStatus::InProgress
                } else {
                    RequestStatus::Error(status_to_string(status))
                }
            } else {
                RequestStatus::Complete
            }
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        unsafe {
            if rust_ucs_ptr_is_ptr(self.request) != 0 {
                ucp_request_free(self.request);
            }
            if let Some(data) = self.data {
                let _ = Box::from_raw(data);
            }
        }
    }
}

/// Request data struct used to hold callback user data for a request.
pub(crate) struct RequestData {
    /// Request boolean set in the callback.
    complete: bool,

    /// Wraps created datatype and extra context info.
    buffer: UCXBuffer,
}

impl RequestData {
    /// Create a new request data struct for callback user data.
    pub fn new(buffer: UCXBuffer) -> RequestData {
        RequestData {
            complete: false,
            buffer,
        }
    }
}

/// This function is invoked when the status changes for a non-blocking send
/// request.
pub(crate) unsafe extern "C" fn send_nbx_callback(
    _req: *mut c_void,
    status: ucs_status_t,
    user_data: *mut c_void,
) {
    let req_data = user_data as *mut RequestData;
    (*req_data).complete = status == UCS_OK;
}

/// This function is invoked when the status changes for a non-blocking receive
/// request.
pub(crate) unsafe extern "C" fn tag_recv_nbx_callback(
    _req: *mut c_void,
    status: ucs_status_t,
    _tag_info: *const ucp_tag_recv_info_t,
    user_data: *mut c_void,
) {
    let req_data = user_data as *mut RequestData;
    (*req_data).complete = status == UCS_OK;
}
