//! Various callback functions needed for UCP.
use mpicd_ucx_sys::{ucp_tag_recv_info_t, ucs_status_t, UCS_OK};
use std::os::raw::c_void;

/// This function is invoked when the status changes for a non-blocking send
/// request.
pub(crate) unsafe extern "C" fn send_nbx_callback(
    _req: *mut c_void,
    status: ucs_status_t,
    user_data: *mut c_void,
) {
    let cb_info = user_data as *mut bool;
    *cb_info = status == UCS_OK;
}

/// This function is invoked when the status changes for a non-blocking receive
/// request.
pub(crate) unsafe extern "C" fn tag_recv_nbx_callback(
    _req: *mut c_void,
    status: ucs_status_t,
    _tag_info: *const ucp_tag_recv_info_t,
    user_data: *mut c_void,
) {
    let done = user_data as *mut bool;
    *done = status == UCS_OK;
}
