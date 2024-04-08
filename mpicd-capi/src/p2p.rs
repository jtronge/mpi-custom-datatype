use crate::datatype::Datatype;
use crate::{consts, with_context, Comm, Request, Status};
use std::ffi::{c_int, c_void};
use mpicd::communicator::{Communicator, Message, MessageMut};

struct SendBuffer {
    ptr: *const u8,
    size: usize,
}

impl Message for SendBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Send(buf: *const c_void, count: c_int, datatype: Datatype, dest: c_int, tag: c_int, _comm: Comm) -> c_int {
    with_context(move |ctx| {
        let send_buffer = SendBuffer {
            ptr: buf as *const _,
            size: count.try_into().unwrap(),
        };
        let req = ctx.isend(send_buffer, dest, tag).expect("failed to send request");
        let _ = ctx.waitall(&[req]);
        consts::SUCCESS
    })
}

struct RecvBuffer {
    ptr: *mut u8,
    size: usize,
}

impl MessageMut for RecvBuffer {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Recv(buf: *mut c_void, count: c_int, datatype: Datatype, source: c_int, tag: c_int, _comm: Comm) -> c_int {
    with_context(move |ctx| {
        let recv_buffer = RecvBuffer {
            ptr: buf as *mut _,
            size: count.try_into().unwrap(),
        };
        let req = ctx.irecv(recv_buffer, source, tag).expect("failed to receive request");
        let _ = ctx.waitall(&[req]);
        consts::SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Isend(
    buf: *const c_void,
    count: c_int,
    datatype: Datatype,
    dest: c_int,
    tag: c_int,
    _comm: Comm,
    request: *mut Request,
) -> c_int {
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Irecv(
    buf: *mut c_void,
    count: c_int,
    datatype: Datatype,
    source: c_int,
    tag: c_int,
    _comm: Comm,
    request: *mut Request,
) -> c_int {
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Waitall(
    count: c_int,
    array_of_requests: *mut Request,
    array_of_statuses: *mut Status,
) -> c_int {
    consts::SUCCESS
}
