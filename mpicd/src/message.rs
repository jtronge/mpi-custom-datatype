//! Request object.
use std::ffi::c_void;
use mpicd_ucx_sys::{
    rust_ucs_ptr_is_err, rust_ucs_ptr_is_ptr, rust_ucs_ptr_status, rust_ucp_dt_make_contig,
    rust_ucp_dt_make_iov, ucp_dt_iov_t, ucp_tag_recv_info_t, ucp_request_free, ucs_status_t,
    ucp_worker_progress, UCS_OK, UCS_INPROGRESS,
};
use crate::{status_to_string, Status, System};
use crate::request::Request;
use crate::datatype::{PackMethod, UnpackMethod};

const BUFSIZE: usize = 8192;

pub trait Message {
    /// Progress the message and return the status.
    unsafe fn progress(&mut self, system: &mut System) -> Status;
}

pub(crate) struct PackSendMessage {
    /// Pack method.
    pack_method: Box<dyn PackMethod>,

    /// Destination rank.
    dest: usize,

    /// Message tag.
    tag: u64,

    /// Packed message buffer.
    packed_buffer: Vec<u8>,

    /// Offset in the packed buffer.
    offset: usize,

    /// Iovec send data.
    iovdata: Option<Vec<ucp_dt_iov_t>>,

    /// Pending request.
    req: Option<Request>,
}

impl PackSendMessage {
    pub(crate) unsafe fn new(pack_method: Box<dyn PackMethod>, dest: i32, tag: u64) -> PackSendMessage {
        // Allocate the packed buffer if necessary.
        let packed_buffer = match pack_method.packed_size() {
            Ok(size) => vec![0; size],
            Err(err) => panic!("Error occured while getting the packed size of a type: {:?}", err),
        };

        PackSendMessage {
            pack_method,
            dest: dest as usize,
            tag,
            packed_buffer,
            offset: 0,
            iovdata: None,
            req: None,
        }
    }
}

impl Message for PackSendMessage {
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        if let Some(req) = self.req.as_ref() {
            // Already have a request, just need to wait on it.
            ucp_worker_progress(system.worker);
            req.status()
        } else if self.offset < self.packed_buffer.len() {
            // Pack the buffer in BUFSIZE units.
            let remain = self.packed_buffer.len() - self.offset;
            let dst_size = if BUFSIZE < remain { BUFSIZE } else { remain };
            let dst = self.packed_buffer.as_mut_ptr().offset(self.offset as isize);
            let used = self.pack_method
                .pack(self.offset, dst, dst_size)
                .expect("failed to pack buffer");
            assert!(used <= dst_size);
            self.offset += used;

            // Waiting on packing part.
            Status::InProgress
        } else {
            // Need to get the iovec data and submit the request.
            let regions = self.pack_method
                .memory_regions()
                .expect("failed to get memory regions for type");

            let mut iovdata = vec![];
            // TODO: Must be careful about moving the data. Perhaps this
            // should be Pinned in some way?
            if self.packed_buffer.len() > 0 {
                iovdata.push(ucp_dt_iov_t {
                    buffer: self.packed_buffer.as_mut_ptr() as *mut _,
                    length: self.packed_buffer.len(),
                });
            }

            for (buffer, length) in regions {
                iovdata.push(ucp_dt_iov_t {
                    buffer: buffer as *mut _,
                    length: length,
                });
            }

            let count = iovdata.len();
            let _ = self.iovdata.insert(iovdata);

            // Submit the request with both packed and memory region data.
            let _ = self.req.insert(Request::send_nb(
                system.endpoints[self.dest],
                self.iovdata.as_ref().expect("missing iovec data").as_ptr() as *const _,
                count,
                rust_ucp_dt_make_iov(),
                self.tag,
            ));
            Status::InProgress
        }
    }
}

pub(crate) struct PackRecvMessage {
    /// Pack method.
    unpack_method: Box<dyn UnpackMethod>,

    /// Message tag.
    tag: u64,

    /// Packed message buffer.
    packed_buffer: Vec<u8>,

    /// Iovec receive data.
    iovdata: Option<Vec<ucp_dt_iov_t>>,

    /// Pending request.
    req: Option<Request>,
}

impl PackRecvMessage {
    pub(crate) unsafe fn new(unpack_method: Box<dyn UnpackMethod>, tag: u64) -> PackRecvMessage {
        // Allocate the packed buffer if necessary.
        let packed_buffer = match unpack_method.packed_size() {
            Ok(size) => vec![0; size],
            Err(err) => panic!("Error occured while getting the packed size of a type: {:?}", err),
        };

        PackRecvMessage {
            unpack_method,
            tag,
            packed_buffer,
            iovdata: None,
            req: None,
        }
    }
}

impl Message for PackRecvMessage {
    unsafe fn progress(&mut self, system: &mut System) -> Status {
        if let Some(req) = self.req.as_ref() {
            // Progress the existing request.
            ucp_worker_progress(system.worker);
            match req.status() {
                Status::Complete => {
                    if self.packed_buffer.len() > 0 {
                        // Now need to unpack the data.
                        self.unpack_method
                            .unpack(0, self.packed_buffer.as_ptr(), self.packed_buffer.len())
                            .expect("failed to unpack the data");
                    }
                    Status::Complete
                }
                status => status,
            }
        } else {
            // Need to get the iovec data and submit the request.
            let regions = self.unpack_method
                .memory_regions()
                .expect("failed to get memory regions for type");

            let mut iovdata = vec![];
            // TODO: Must be careful about moving the data. Perhaps this
            // should be Pinned in some way?
            if self.packed_buffer.len() > 0 {
                iovdata.push(ucp_dt_iov_t {
                    buffer: self.packed_buffer.as_mut_ptr() as *mut _,
                    length: self.packed_buffer.len(),
                });
            }

            for (buffer, length) in regions {
                iovdata.push(ucp_dt_iov_t {
                    buffer: *buffer as *mut _,
                    length: length,
                });
            }
            let count = iovdata.len();
            let _ = self.iovdata.insert(iovdata);

            let _ = self.req.insert(Request::recv_nb(
                system.worker,
                self.iovdata.as_ref().expect("missing iovec data").as_ptr() as *mut _,
                count,
                rust_ucp_dt_make_iov(),
                self.tag,
            ));
            Status::InProgress
        }
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
