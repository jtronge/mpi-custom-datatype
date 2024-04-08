//! UCX context handle
use crate::{
    callbacks,
    communicator::{self, Communicator, Message, MessageMut},
    Handle, Request, RequestStatus,
};
use mpicd_ucx_sys::{
    rust_ucp_dt_make_contig, ucp_request_param_t,
    ucp_request_param_t__bindgen_ty_1, ucp_tag_recv_nbx, ucp_tag_send_nbx,
    ucp_worker_progress, UCP_OP_ATTR_FIELD_CALLBACK, UCP_OP_ATTR_FIELD_DATATYPE,
    UCP_OP_ATTR_FIELD_USER_DATA, UCP_OP_ATTR_FLAG_NO_IMM_CMPL,

};
use std::cell::RefCell;
use std::rc::Rc;

pub struct Context {
    /// Handle with ucx info.
    handle: Rc<RefCell<Handle>>,
}

impl Context {
    /// Create a new context.
    pub(crate) fn new(handle: Rc<RefCell<Handle>>) -> Context {
        Context { handle }
    }
}

impl Communicator for Context {
    type Request = usize;

    fn size(&self) -> i32 {
        self.handle.borrow().size as i32
    }

    fn rank(&self) -> i32 {
        self.handle.borrow().rank as i32
    }

    unsafe fn isend<M: Message>(
        &self,
        data: M,
        dest: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        let mut handle = self.handle.borrow_mut();
        assert!(dest < (handle.size as i32));
        let endpoint = handle.endpoints[dest as usize].clone();
        let datatype = rust_ucp_dt_make_contig(1) as u64;
        // Callback info
        let cb_info: *mut bool = Box::into_raw(Box::new(false));
        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA,
            datatype,
            cb: ucp_request_param_t__bindgen_ty_1 {
                send: Some(callbacks::send_nbx_callback),
            },
            user_data: cb_info as *mut _,
            ..Default::default()
        };

        let request = ucp_tag_send_nbx(
            endpoint,
            data.as_ptr() as *const _,
            data.count(),
            encode_tag(dest, tag),
            &param,
        );

        let req_id = handle.add_request(Request {
            request,
            complete: Some(cb_info),
        });
        Ok(req_id)
    }

    unsafe fn irecv<M: MessageMut>(
        &self,
        mut data: M,
        source: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        let mut handle = self.handle.borrow_mut();
        assert!(source < (handle.size as i32));
        let datatype = rust_ucp_dt_make_contig(1) as u64;
        // Callback info
        let cb_info: *mut bool = Box::into_raw(Box::new(false));
        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA
                | UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
            datatype,
            cb: ucp_request_param_t__bindgen_ty_1 {
                recv: Some(callbacks::tag_recv_nbx_callback),
            },
            user_data: cb_info as *mut _,
            ..Default::default()
        };

        let request = ucp_tag_recv_nbx(
            handle.worker,
            data.as_mut_ptr() as *mut _,
            data.count(),
            encode_tag(source, tag),
            TAG_MASK,
            &param,
        );

        // TODO: How do you determine that this is coming from the right rank?
        let req_id = handle.add_request(Request {
            request,
            complete: Some(cb_info),
        });
        Ok(req_id)
    }

    /// Wait for all requests to complete.
    unsafe fn waitall(
        &self,
        requests: &[Self::Request],
    ) -> communicator::Result<Vec<RequestStatus>> {
        let mut handle = self.handle.borrow_mut();
        let mut statuses = vec![RequestStatus::InProgress; requests.len()];
        let mut complete = 0;
        while complete < requests.len() {
            for _ in 0..requests.len() {
                ucp_worker_progress(handle.worker);
            }

            for (i, req) in requests.iter().enumerate() {
                match statuses[i] {
                    RequestStatus::InProgress => (),
                    _ => continue,
                }

                match handle.request_status(*req) {
                    RequestStatus::InProgress => (),
                    status => {
                        statuses[i] = status;
                        handle.remove_request(*req);
                        complete += 1;
                    }
                }
            }
        }
        Ok(statuses)
    }
}

const TAG_MASK: u64 = !0;

/// Encode a tag into a 64-bit UCX tag.
#[inline]
fn encode_tag(rank: i32, tag: i32) -> u64 {
    let rank = rank as u64;
    let tag = tag as u64;
    rank << 32 & tag
}
