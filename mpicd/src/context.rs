//! Context handle code for an MPI application.
use crate::{
    communicator::{self, Communicator},
    datatype::{Buffer, UCXBuffer},
    request::{self, Request, RequestStatus, RequestData},
    Handle,
};
use mpicd_ucx_sys::{
    rust_ucp_dt_make_contig, ucp_request_param_t,
    ucp_request_param_t__bindgen_ty_1, ucp_tag_recv_nbx, ucp_tag_send_nbx,
    ucp_worker_progress, ucp_tag_recv_info_t, ucs_status_t,
    UCP_OP_ATTR_FIELD_CALLBACK, UCP_OP_ATTR_FIELD_DATATYPE,
    UCP_OP_ATTR_FIELD_USER_DATA, UCP_OP_ATTR_FLAG_NO_IMM_CMPL, UCS_OK,
};
use std::os::raw::c_void;
use std::cell::RefCell;
use std::rc::Rc;
use log::debug;

/// Context handle.
///
/// This implements Communicator and also acts as MPI_COMM_WORLD would in a
/// standard MPI application.
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

    unsafe fn isend<B: Buffer>(
        &self,
        data: B,
        dest: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        let mut handle = self.handle.borrow_mut();
        assert!(dest < (handle.size as i32));
        let endpoint = handle.endpoints[dest as usize].clone();
        // let datatype = rust_ucp_dt_make_contig(1) as u64;

        let ptr = data.as_ptr();
        let count = data.count();
        let datatype = UCXBuffer::new_type(&data, ptr, None, count);
        let dt_id = datatype.dt_id();
        let ptr = datatype.buf_ptr();
        let count = datatype.buf_count();
        let req_data: *mut RequestData = Box::into_raw(Box::new(RequestData::new(datatype)));

        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA,
            datatype: dt_id,
            cb: ucp_request_param_t__bindgen_ty_1 {
                send: Some(request::send_nbx_callback),
            },
            user_data: req_data as *mut _,
            ..Default::default()
        };

        let request = ucp_tag_send_nbx(
            endpoint,
            ptr,
            count,
            encode_tag(dest, tag),
            &param,
        );

        let req_id = handle.add_request(Request::new(request, Some(req_data)));
        Ok(req_id)
    }

    unsafe fn irecv<B: Buffer>(
        &self,
        mut data: B,
        source: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        let mut handle = self.handle.borrow_mut();
        assert!(source < (handle.size as i32));
        // let datatype = rust_ucp_dt_make_contig(1) as u64;

        let ptr = data.as_ptr();
        let ptr_mut = data.as_mut_ptr();
        let count = data.count();
        let datatype = UCXBuffer::new_type(&mut data, ptr, ptr_mut, count);
        let dt_id = datatype.dt_id();
        let ptr = datatype.buf_ptr_mut().expect("missing mutable buffer pointer");
        let count = datatype.buf_count();

        // Callback info
        let req_data: *mut RequestData = Box::into_raw(Box::new(RequestData::new(datatype)));
        let param = ucp_request_param_t {
            op_attr_mask: UCP_OP_ATTR_FIELD_DATATYPE
                | UCP_OP_ATTR_FIELD_CALLBACK
                | UCP_OP_ATTR_FIELD_USER_DATA
                | UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
            datatype: dt_id,
            cb: ucp_request_param_t__bindgen_ty_1 {
                recv: Some(request::tag_recv_nbx_callback),
            },
            user_data: req_data as *mut _,
            ..Default::default()
        };

        debug!("(receive call) data.count() = {}", data.count());

        // NOTE: The correct source rank is encoded in the tag.
        let request = ucp_tag_recv_nbx(
            handle.worker,
            ptr,
            count,
            encode_tag(source, tag),
            TAG_MASK,
            &param,
        );

        let req_id = handle.add_request(Request::new(request, Some(req_data)));
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
                if statuses[i] != RequestStatus::InProgress {
                    continue;
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

/// The tag mask used for receive requests; for now all bits are important.
const TAG_MASK: u64 = !0;

/// Encode a tag into a 64-bit UCX tag.
#[inline]
fn encode_tag(rank: i32, tag: i32) -> u64 {
    let rank = rank as u64;
    let tag = tag as u64;
    rank << 32 & tag
}
