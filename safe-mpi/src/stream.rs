use log::info;
use std::io::{Read, Write, Result};
use std::mem::MaybeUninit;
use std::os::raw::c_void;
use ucx2_sys::{
    ucp_ep_h,
    ucp_request_param_t,
    ucp_worker_h,
    ucp_worker_progress,
    rust_ucp_dt_make_contig,
    rust_ucs_ptr_is_ptr,
    rust_ucs_ptr_is_err,
    rust_ucs_ptr_status,
    ucp_stream_recv_nbx,
    ucp_stream_send_nbx,
    UCP_OP_ATTR_FIELD_DATATYPE,
    UCP_OP_ATTR_FIELD_CALLBACK,
    UCP_OP_ATTR_FIELD_USER_DATA,
    ucs_status_t,
    UCS_OK,
    UCS_INPROGRESS,
};
use crate::status_to_string;

/// The Stream struct wraps ucp streams, giving it a Read and Write interface.
pub(crate) struct Stream {
    worker: ucp_worker_h,
    endpoint: ucp_ep_h,
    reqs: Vec<*mut c_void>,
}

impl Stream {
    pub(crate) fn new(worker: ucp_worker_h, endpoint: ucp_ep_h) -> Stream {
        Stream {
            worker,
            endpoint,
            reqs: vec![],
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        unsafe {
            let mut length = 0;
            let mut param = MaybeUninit::<ucp_request_param_t>::uninit().assume_init();
            param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE
                                 | UCP_OP_ATTR_FIELD_CALLBACK
                                 | UCP_OP_ATTR_FIELD_USER_DATA;
            param.datatype = rust_ucp_dt_make_contig(buf.len()).try_into().unwrap();
            param.cb.recv_stream = Some(stream_recv_nbx_callback);
            // Allocate callback info
            let cb_info: Box<Option<usize>> = Box::new(None);
            let cb_info = Box::into_raw(cb_info);
            param.user_data = cb_info as *mut _;

            let req = ucp_stream_recv_nbx(
                self.endpoint,
                buf.as_ptr() as *mut _,
                buf.len() * std::mem::size_of::<u8>(),
                &mut length,
                &param,
            );

            // Check for immediate completion
            let res = if req == std::ptr::null_mut() {
                info!("length as set: {}", length);
                // Ok(length)
                // ucp bug?
                Ok(buf.len())
            } else {
                wait_loop(self.worker, req, || (*cb_info).is_some());
                let length = (*cb_info).unwrap();
                if length > buf.len() {
                    Ok(buf.len())
                } else {
                    Ok(length)
                }
            };

            info!("read result: {:?}, buf.len(): {}", res, buf.len());
            // Deallocate callback info
            let _ = Box::from_raw(cb_info);
            res
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        unsafe {
            let mut param = MaybeUninit::<ucp_request_param_t>::uninit().assume_init();
            param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE
                                 | UCP_OP_ATTR_FIELD_CALLBACK
                                 | UCP_OP_ATTR_FIELD_USER_DATA;
            param.datatype = rust_ucp_dt_make_contig(buf.len()).try_into().unwrap();
            param.cb.send = Some(send_nbx_callback);
            // Allocate callback info
            let cb_info: *mut bool = Box::into_raw(Box::new(false));
            param.user_data = cb_info as *mut _;

            let req = ucp_stream_send_nbx(
                self.endpoint,
                buf.as_ptr() as *const _,
                buf.len() * std::mem::size_of::<u8>(),
                &param,
            );

            info!("wrote buf.len(): {}", buf.len());
            wait_loop(self.worker, req, || *cb_info);

            // Deallocate the callback info
            let _ = Box::from_raw(cb_info);
            Ok(buf.len())
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Wait for the request to complete
unsafe fn wait_loop<F>(worker: ucp_worker_h, req: *mut c_void, f: F)
where
    F: Fn() -> bool,
{
    // TODO: Maybe this check should be done in the calling code
    if rust_ucs_ptr_is_ptr(req) == 0 {
        let status = rust_ucs_ptr_status(req);
        if status != UCS_OK {
            panic!("Request failed: {}", status_to_string(status));
        }
        // Already complete
        return;
    }

    if rust_ucs_ptr_is_err(req) != 0 {
        panic!("Failed to send data");
    }

    while !f() {
        info!("Waiting for request completion");
        for _ in 0..512 {
            ucp_worker_progress(worker);
        }

        let status = rust_ucs_ptr_status(req);
        if status != UCS_INPROGRESS {
            if status != UCS_OK {
                panic!("Failed request: {}", status_to_string(status));
            }
            break;
        }
    }
}

unsafe extern "C" fn stream_recv_nbx_callback(
    req: *mut c_void,
    status: ucs_status_t,
    length: usize,
    user_data: *mut c_void,
) {
    if status == UCS_OK {
        let cb_info = user_data as *mut Option<usize>;
        info!("length: {}", length);
        (*cb_info).insert(length);
    }
}

unsafe extern "C" fn send_nbx_callback(
    req: *mut c_void,
    status: ucs_status_t,
    user_data: *mut c_void,
) {
    if status == UCS_OK {
        let cb_info = user_data as *mut bool;
        *cb_info = true;
    }
}
