use mpicd::{
    communicator::Communicator,
};
use std::ffi::{c_int, c_void};
use crate::{
    datatype::{BufferPointer, CustomBuffer, ByteBuffer},
    c, consts, with_context,
};

#[no_mangle]
pub unsafe extern "C" fn MPI_Send(
    buf: *const c_void,
    count: c_int,
    datatype: c::Datatype,
    dest: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::ReturnStatus {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx, cctx| {
        let req = if let Some(custom_datatype) = cctx.get_custom_datatype(datatype) {
            let buffer = CustomBuffer {
                ptr: BufferPointer::Const(buf as *const _),
                len: count as usize,
                custom_datatype,
            };
            ctx
                .isend(buffer, dest, tag)
                .expect("failed to send request")
        } else {
            // Assume MPI_BYTE
            assert_eq!(datatype, consts::BYTE);

            let buffer = ByteBuffer {
                ptr: BufferPointer::Const(buf as *const _),
                size: count.try_into().unwrap(),
            };
            ctx
                .isend(buffer, dest, tag)
                .expect("failed to send request")
        };

        let _ = ctx.waitall(&[req]);
        consts::SUCCESS
    })
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Recv(
    buf: *mut c_void,
    count: c_int,
    datatype: c::Datatype,
    source: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::ReturnStatus {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx, cctx| {
        let req = if let Some(custom_datatype) = cctx.get_custom_datatype(datatype) {
            let buffer = CustomBuffer {
                ptr: BufferPointer::Mut(buf as *mut _),
                len: count as usize,
                custom_datatype,
            };
            ctx
                .irecv(buffer, source, tag)
                .expect("failed to receive request")
        } else {
            // Assume MPI_BYTE
            assert_eq!(datatype, consts::BYTE);

            let buffer = ByteBuffer {
                ptr: BufferPointer::Mut(buf as *mut _),
                size: count.try_into().unwrap(),
            };
            ctx
                .irecv(buffer, source, tag)
                .expect("failed to receive request")
        };

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
