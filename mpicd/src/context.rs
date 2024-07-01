//! Context handle code for an MPI application.
use crate::{
    communicator::{self, Communicator},
    datatype::MessageBuffer,
    message::{PackSendMessage, PackRecvMessage, ContiguousSendMessage, ContiguousRecvMessage},
    request::{encode_tag, decode_tag, BARRIER_TAG, PROBE_TAG_MASK, TAG_MASK},
    Handle, Status,
};
use mpicd_ucx_sys::{ucp_tag_probe_nb, ucp_worker_progress};
use std::cell::RefCell;
use std::mem::MaybeUninit;
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

        if let Some(packer) = data.pack() {
            let packer = packer.expect("failed to initialize PackState");
            let request = PackSendMessage::new(packer, dest, tag);
            Ok(handle.add_message(request))
        } else {
            let request = ContiguousSendMessage::new(data.ptr() as *const _, data.count(), dest, tag);
            Ok(handle.add_message(request))
        }
    }

    unsafe fn internal_irecv<B: MessageBuffer + ?Sized>(
        &self,
        data: &mut B,
        tag: u64,
    ) -> communicator::Result<<Self as Communicator>::Request> {
        let mut handle = self.handle.borrow_mut();

        if let Some(unpack_method) = data.unpack() {
            let unpack_method = unpack_method
                .expect("failed to initialize pack method");
            let request = PackRecvMessage::new(unpack_method, tag);
            Ok(handle.add_message(request))
        } else {
            let request = ContiguousRecvMessage::new(data.ptr_mut(), data.count(), tag);
            Ok(handle.add_message(request))
        }
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

    fn probe(&self, source: Option<i32>, tag: i32) -> communicator::Result<communicator::ProbeResult> {
        unsafe {
            let mut info = MaybeUninit::uninit();
            let handle = self.handle.borrow_mut();
            let (tag, tag_mask) = if let Some(source) = source {
                (encode_tag(0, source, tag), TAG_MASK)
            } else {
                (encode_tag(0, 0, tag), PROBE_TAG_MASK)
            };
            let result = ucp_tag_probe_nb(handle.system.worker, tag, tag_mask, 0, info.as_mut_ptr());
            if result != std::ptr::null_mut() {
                let info = info.assume_init();
                let (_, source, _) = decode_tag(info.sender_tag);
                Ok(communicator::ProbeResult {
                    size: info.length,
                    source,
                })
            } else {
                Err(communicator::Error::NoProbeMessage)
            }
        }
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
