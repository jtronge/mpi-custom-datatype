//! Point-to-point utility functions.
use std::ffi::c_void;
use mpicd_ucx_sys::{
    rust_ucs_ptr_is_ptr, rust_ucs_ptr_is_err, rust_ucs_ptr_status,
    ucs_status_t, ucs_status_ptr_t, ucp_ep_h, ucp_worker_h, ucp_datatype_t, ucp_request_param_t,
    ucp_request_param_t__bindgen_ty_1, ucp_tag_send_nbx, ucp_tag_recv_nbx,
    ucp_tag_recv_info_t, ucp_request_free, UCP_OP_ATTR_FIELD_DATATYPE, UCP_OP_ATTR_FIELD_CALLBACK,
    UCP_OP_ATTR_FIELD_USER_DATA, UCP_OP_ATTR_FLAG_NO_IMM_CMPL, UCS_OK, UCS_INPROGRESS,
};
use crate::{Status, status_to_string};

pub(crate) struct Request {
    pub(crate) req: ucs_status_ptr_t,

    pub(crate) req_data: *mut RequestData,
}

impl Request {
    /// Initiate a non-blocking send and return the ucx request.
    pub(crate) unsafe fn send_nb(
        endpoint: ucp_ep_h,
        ptr: *const u8,
        count: usize,
        datatype: ucp_datatype_t,
        tag: u64,
    ) -> Request {
        let req_data: *mut RequestData = Box::into_raw(Box::new(RequestData::new(datatype)));
        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA,
            datatype,
            cb: ucp_request_param_t__bindgen_ty_1 {
                send: Some(send_nbx_callback),
            },
            user_data: req_data as *mut _,
            ..Default::default()
        };

        let req = ucp_tag_send_nbx(
            endpoint,
            ptr as *const _,
            count,
            tag,
            &param,
        );

        Request {
            req,
            req_data,
        }
    }

    /// Initiate a non-blocking receive and return the ucx request.
    pub(crate) unsafe fn recv_nb(
        worker: ucp_worker_h,
        ptr: *mut u8,
        count: usize,
        datatype: ucp_datatype_t,
        tag: u64,
    ) -> Request {
        let req_data: *mut RequestData = Box::into_raw(Box::new(RequestData::new(datatype)));
        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA
                | UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
            datatype,
            cb: ucp_request_param_t__bindgen_ty_1 {
                recv: Some(tag_recv_nbx_callback),
            },
            user_data: req_data as *mut _,
            ..Default::default()
        };

        // debug!("(receive call) data.count() = {}", data.count());

        // NOTE: The correct source rank is encoded in the tag.
        let req = ucp_tag_recv_nbx(
            worker,
            ptr as *mut _,
            count,
            tag,
            TAG_MASK,
            &param,
        );

        Request {
            req,
            req_data,
        }
    }

    pub(crate) unsafe fn status(&self) -> Status {
        if rust_ucs_ptr_is_ptr(self.req) == 0 {
            let status = rust_ucs_ptr_status(self.req);
            if status != UCS_OK {
                Status::Error(status_to_string(status))
            } else {
                Status::Complete
            }
        } else if rust_ucs_ptr_is_err(self.req) != 0 {
            Status::Error("Internal pointer failure".to_string())
        } else {
            let status = rust_ucs_ptr_status(self.req);
            if !(*self.req_data.as_ref().unwrap()).complete {
                if status == UCS_OK {
                    Status::Complete
                } else if status == UCS_INPROGRESS {
                    Status::InProgress
                } else {
                    Status::Error(status_to_string(status))
                }
            } else {
                Status::Complete
            }
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        unsafe {
            if rust_ucs_ptr_is_ptr(self.req) != 0 {
                ucp_request_free(self.req);
            }
            let _ = Box::from_raw(self.req_data);
        }
    }
}

/// The tag mask used for receive requests; for now all bits are important.
const TAG_MASK: u64 = !0;

/// Request data struct used to hold callback user data for a request.
pub(crate) struct RequestData {
    /// Request boolean set in the callback.
    complete: bool,

    /// Hold onto any datatypes created until completion of request.
    _datatype: ucp_datatype_t,
}

impl RequestData {
    /// Create a new request data struct for callback user data.
    pub fn new(datatype: ucp_datatype_t) -> RequestData {
        RequestData {
            complete: false,
            _datatype: datatype,
        }
    }
}

/// This function can be invoked when the status changes for a non-blocking send
/// request.
pub(crate) unsafe extern "C" fn send_nbx_callback(
    _req: *mut c_void,
    status: ucs_status_t,
    user_data: *mut c_void,
) {
    let req_data = user_data as *mut RequestData;
    (*req_data).complete = status == UCS_OK;
}

/// This function can be invoked when the status changes for a non-blocking receive
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
