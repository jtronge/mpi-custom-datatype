//! Request object.
use std::ffi::c_void;
use mpicd_ucx_sys::{
    rust_ucs_ptr_is_err, rust_ucs_ptr_is_ptr, rust_ucs_ptr_status, rust_ucp_dt_make_contig,
    ucp_tag_recv_info_t, ucp_request_free, ucs_status_t, ucp_worker_progress,
    UCS_OK, UCS_INPROGRESS,
};
use crate::{status_to_string, Status, System};
use crate::request::Request;
use crate::datatype::PackMethod;

pub trait Message {
    /// Progress the message and return the status.
    unsafe fn progress(&mut self, system: &mut System) -> Status;
}

pub(crate) struct PackSendMessage {
    pack_method: Box<dyn PackMethod>,
    dest: i32,
    tag: u64,
}

impl PackSendMessage {
    pub(crate) fn new(pack_method: Box<dyn PackMethod>, dest: i32, tag: u64) -> PackSendMessage {
        PackSendMessage {
            pack_method,
            dest,
            tag,
        }
    }
}

impl Message for PackSendMessage {
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        // TODO
        Status::InProgress
    }
}

pub(crate) struct PackRecvMessage {
    pack_method: Box<dyn PackMethod>,
    tag: u64,
}

impl PackRecvMessage {
    pub(crate) fn new(pack_method: Box<dyn PackMethod>, tag: u64) -> PackRecvMessage {
        PackRecvMessage {
            pack_method,
            tag,
        }
    }
}

impl Message for PackRecvMessage {
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        // TODO
        Status::InProgress
    }
}

/// Send message for contiguous data.
pub(crate) struct ContiguousSendMessage {
    ptr: *const u8,
    count: usize,
    dest: usize,
    tag: u64,
    req: Option<Request>,
}

impl ContiguousSendMessage {
    pub(crate) fn new(ptr: *const u8, count: usize, dest: i32, tag: u64) -> ContiguousSendMessage {
        ContiguousSendMessage {
            ptr,
            count,
            dest: dest as usize,
            tag,
            req: None,
        }
    }
}

impl Message for ContiguousSendMessage {
    /// Get the status of this message.
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        if let Some(req) = self.req.as_mut() {
            ucp_worker_progress(system.worker);
            req.status()
        } else {
            let _ = self.req.insert(Request::send_nb(
                system.endpoints[self.dest],
                self.ptr,
                self.count,
                rust_ucp_dt_make_contig(1),
                self.tag,
            ));
            Status::InProgress
        }
    }
}

/// Send message for contiguous data.
pub(crate) struct ContiguousRecvMessage {
    ptr: *mut u8,
    count: usize,
    tag: u64,
    req: Option<Request>,
}

impl ContiguousRecvMessage {
    pub(crate) fn new(ptr: *mut u8, count: usize, tag: u64) -> ContiguousRecvMessage {
        ContiguousRecvMessage {
            ptr,
            count,
            tag,
            req: None,
        }
    }
}

impl Message for ContiguousRecvMessage {
    /// Get the status of this message.
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        if let Some(req) = self.req.as_mut() {
            ucp_worker_progress(system.worker);
            req.status()
        } else {
            let _ = self.req.insert(Request::recv_nb(
                system.worker,
                self.ptr,
                self.count,
                rust_ucp_dt_make_contig(1),
                self.tag,
            ));
            Status::InProgress
        }
        // TODO
    }
}
