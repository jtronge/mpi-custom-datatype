//! Request object.
use std::ffi::c_void;
use mpicd_ucx_sys::{
    rust_ucs_ptr_is_err, rust_ucs_ptr_is_ptr, rust_ucs_ptr_status,
    ucp_tag_recv_info_t, ucp_request_free, ucs_status_t, UCS_OK, UCS_INPROGRESS,
};
use crate::{status_to_string, Status, System};
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
    dest: i32,
    tag: u64,
}

impl ContiguousSendMessage {
    pub(crate) fn new(ptr: *const u8, count: usize, dest: i32, tag: u64) -> ContiguousSendMessage {
        ContiguousSendMessage {
            ptr,
            count,
            dest,
            tag,
        }
    }
}

impl Message for ContiguousSendMessage {
    /// Get the status of this message.
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        // TODO
        Status::InProgress
    }
}

/// Send message for contiguous data.
pub(crate) struct ContiguousRecvMessage {
    ptr: *mut u8,
    count: usize,
    tag: u64,
}

impl ContiguousRecvMessage {
    pub(crate) fn new(ptr: *mut u8, count: usize, tag: u64) -> ContiguousRecvMessage {
            ContiguousRecvMessage {
                ptr,
                count,
                tag,
            }
    }
}

impl Message for ContiguousRecvMessage {
    /// Get the status of this message.
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        // TODO
        Status::InProgress
    }
}
