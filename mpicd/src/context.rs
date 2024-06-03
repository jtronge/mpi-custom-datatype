//! Context handle code for an MPI application.
use crate::{
    communicator::{self, Communicator},
    datatype::MessageBuffer,
    message::{PackSendMessage, PackRecvMessage, ContiguousSendMessage, ContiguousRecvMessage},
    Handle, Status,
};
use mpicd_ucx_sys::ucp_worker_progress;
use std::cell::RefCell;
use std::rc::Rc;

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

    unsafe fn internal_isend<B: MessageBuffer + ?Sized>(
        &self,
        data: &B,
        dest: i32,
        tag: u64,
    ) -> communicator::Result<<Self as Communicator>::Request> {
        let mut handle = self.handle.borrow_mut();
        assert!(dest < (handle.system.size as i32));
        // let endpoint = handle.endpoints[dest as usize].clone();
        // let datatype = rust_ucp_dt_make_contig(1) as u64;

        if let Some(packer) = data.pack() {
            let packer = packer.expect("failed to initialize PackState");
            let request = PackSendMessage::new(packer, dest, tag);
            Ok(handle.add_message(request))
        } else {
            let request = ContiguousSendMessage::new(data.ptr() as *const _, data.count(), dest, tag);
            Ok(handle.add_message(request))
        }

/*
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
            data.ptr() as *const _,
            data.count(),
            tag,
            &param,
        );

        let req_id = handle.add_request(Request::new(request, Some(req_data)));
        Ok(req_id)
*/
    }

    unsafe fn internal_irecv<B: MessageBuffer + ?Sized>(
        &self,
        data: &mut B,
        tag: u64,
    ) -> communicator::Result<<Self as Communicator>::Request> {
        let mut handle = self.handle.borrow_mut();
        // let datatype = rust_ucp_dt_make_contig(1) as u64;

        if let Some(unpack_method) = data.unpack() {
            let unpack_method = unpack_method
                .expect("failed to initialize pack method");
            let request = PackRecvMessage::new(unpack_method, tag);
            Ok(handle.add_message(request))
        } else {
            let request = ContiguousRecvMessage::new(data.ptr_mut(), data.count(), tag);
            Ok(handle.add_message(request))
        }

/*
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
            tag,
            TAG_MASK,
            &param,
        );

        let req_id = handle.add_request(Request::new(request, Some(req_data)));
        Ok(req_id)
*/
    }
}

impl Communicator for Context {
    type Request = usize;

    fn size(&self) -> i32 {
        self.handle.borrow().system.size as i32
    }

    fn rank(&self) -> i32 {
        self.handle.borrow().system.rank as i32
    }

    /// Barrier operation on all processes.
    ///
    /// Uses a simple O(n) algorithm.
    fn barrier(&self) {
        unsafe {
            let size = self.handle.borrow().system.size as i32;
            let rank = self.handle.borrow().system.rank as i32;
            if rank == 0 {
                let mut buf = vec![0; 1];
                let mut reqs = vec![];
                for i in 1..size {
                    reqs.push(self.internal_isend(&buf[..], i, encode_tag(BARRIER_TAG, 0, 0)).expect("failed to get send request"));
                }
                self.waitall(&reqs).expect("failed to wait for send requests");

                reqs.clear();
                for i in 1..size {
                    reqs.push(self.internal_irecv(&mut buf[..], encode_tag(BARRIER_TAG, i, 0)).expect("failed to get recv request"));
                }
                self.waitall(&reqs).expect("failed to wait for recv requests");
            } else {
                let mut buf = vec![0; 1];
                let req = self.internal_irecv(&mut buf[..], encode_tag(BARRIER_TAG, 0, 0)).expect("failed to get recv request");
                self.waitall(&[req]).expect("failed to wait for recv request");
                let req = self.internal_isend(&buf[..], 0, encode_tag(BARRIER_TAG, rank, 0)).expect("failed to get send request");
                self.waitall(&[req]).expect("failed to wait for send request");
            }
        }
    }

    unsafe fn isend<B: MessageBuffer + ?Sized>(
        &self,
        data: &B,
        dest: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        let rank = self.handle.borrow().system.rank as i32;
        self.internal_isend(data, dest, encode_tag(0, rank, tag))
    }

    unsafe fn irecv<B: MessageBuffer + ?Sized>(
        &self,
        data: &mut B,
        source: i32,
        tag: i32,
    ) -> communicator::Result<Self::Request> {
        assert!(source < (self.handle.borrow().system.size as i32));
        self.internal_irecv(data, encode_tag(0, source, tag))
    }

    /// Wait for all requests to complete.
    unsafe fn waitall(
        &self,
        requests: &[Self::Request],
    ) -> communicator::Result<Vec<Status>> {
        let mut handle = self.handle.borrow_mut();
        let mut statuses = vec![Status::InProgress; requests.len()];
        let mut complete = 0;

        while complete < requests.len() {
            for (i, req) in requests.iter().enumerate() {
                if statuses[i] != Status::InProgress {
                    continue;
                }

                match handle.message_progress(*req) {
                    Status::InProgress => (),
                    status => {
                        statuses[i] = status;
                        handle.remove_message(*req);
                        complete += 1;
                    }
                }

                ucp_worker_progress(handle.system.worker);
            }
        }
        Ok(statuses)
    }
}

/// Internal tag to be used for barriers.
const BARRIER_TAG: u8 = 1;

/// Encode a tag into a 64-bit UCX tag.
#[inline]
fn encode_tag(internal_tag: u8, rank: i32, tag: i32) -> u64 {
    // The rank should be able to fit into 24 bits.
    assert!(rank < ((1 << 24) - 1));
    let internal_tag = internal_tag as u64;
    let rank = (rank as u64) & 0xFFFFFF;
    let tag = tag as u64;
    (internal_tag << 56) | (rank << 32) | tag
}
