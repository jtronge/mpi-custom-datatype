use crate::{c, consts, with_context};
use mpicd::{
    communicator::Communicator,
    datatype::{SendDatatype, SendKind, RecvDatatype, RecvKind},
};
use std::ffi::{c_int, c_void};

struct SendBuffer {
    ptr: *const u8,
    size: usize,
}

impl SendDatatype for SendBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }

    fn kind(&self) -> SendKind {
        SendKind::Contiguous
    }
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Send(
    buf: *const c_void,
    count: c_int,
    _datatype: c::Datatype,
    dest: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::ReturnStatus {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx| {
        let send_buffer = SendBuffer {
            ptr: buf as *const _,
            size: count.try_into().unwrap(),
        };
        let req = ctx
            .isend(send_buffer, dest, tag)
            .expect("failed to send request");
        let _ = ctx.waitall(&[req]);
        consts::SUCCESS
    })
}

struct RecvBuffer {
    ptr: *mut u8,
    size: usize,
}

impl RecvDatatype for RecvBuffer {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }

    fn kind(&mut self) -> RecvKind {
        RecvKind::Contiguous
    }
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Recv(
    buf: *mut c_void,
    count: c_int,
    _datatype: c::Datatype,
    source: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::ReturnStatus {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx| {
        let recv_buffer = RecvBuffer {
            ptr: buf as *mut _,
            size: count.try_into().unwrap(),
        };
        let req = ctx
            .irecv(recv_buffer, source, tag)
            .expect("failed to receive request");
        let _ = ctx.waitall(&[req]);
        consts::SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Isend(
    _buf: *const c_void,
    _count: c_int,
    _datatype: c::Datatype,
    _dest: c_int,
    _tag: c_int,
    _comm: c::Comm,
    _request: *mut c::Request,
) -> c::ReturnStatus {
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Irecv(
    _buf: *mut c_void,
    _count: c_int,
    _datatype: c::Datatype,
    _source: c_int,
    _tag: c_int,
    _comm: c::Comm,
    _request: *mut c::Request,
) -> c::ReturnStatus {
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Waitall(
    _count: c_int,
    _array_of_requests: *mut c::Request,
    _array_of_statuses: *mut c::Status,
) -> c::ReturnStatus {
    consts::SUCCESS
}
