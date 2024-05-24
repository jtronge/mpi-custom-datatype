use mpicd::{
    communicator::Communicator,
};
use std::ffi::{c_int, c_void};
use crate::{
    datatype::{CustomBuffer, ByteBuffer},
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
    let req = isend(buf, count, datatype, dest, tag, comm);
    with_context(move |ctx, _cctx| {
        let _ = ctx.waitall(&[req.try_into().unwrap()]);
        consts::SUCCESS
    })
}

unsafe fn isend(
    buf: *const c_void,
    count: c_int,
    datatype: c::Datatype,
    dest: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::Request {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx, cctx| {
        let req = if let Some(custom_datatype) = cctx.get_custom_datatype(datatype) {
            let buffer = CustomBuffer {
                ptr: buf as *mut _,
                len: count as usize,
                custom_datatype,
            };
            ctx
                .isend(&buffer, dest, tag)
                .expect("failed to send request")
        } else {
            // Assume MPI_BYTE
            assert_eq!(datatype, consts::BYTE);

            let buffer = ByteBuffer {
                ptr: buf as *mut _,
                size: count.try_into().unwrap(),
            };
            ctx
                .isend(&buffer, dest, tag)
                .expect("failed to send request")
        };

        req.try_into().unwrap()
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
    let req = irecv(buf, count, datatype, source, tag, comm);
    with_context(move |ctx, _cctx| {
        let _ = ctx.waitall(&[req.try_into().unwrap()]);
        consts::SUCCESS
    })
}


unsafe fn irecv(
    buf: *mut c_void,
    count: c_int,
    datatype: c::Datatype,
    source: c_int,
    tag: c_int,
    comm: c::Comm,
) -> c::Request {
    assert_eq!(comm, consts::COMM_WORLD);

    with_context(move |ctx, cctx| {
        let req = if let Some(custom_datatype) = cctx.get_custom_datatype(datatype) {
            let mut buffer = CustomBuffer {
                ptr: buf as *mut _,
                len: count as usize,
                custom_datatype,
            };
            ctx
                .irecv(&mut buffer, source, tag)
                .expect("failed to receive request")
        } else {
            // Assume MPI_BYTE
            assert_eq!(datatype, consts::BYTE);

            let mut buffer = ByteBuffer {
                ptr: buf as *mut _,
                size: count.try_into().unwrap(),
            };
            ctx
                .irecv(&mut buffer, source, tag)
                .expect("failed to receive request")
        };

        req.try_into().expect("failed to convert to isize")
    })
}


#[no_mangle]
pub unsafe extern "C" fn MPI_Isend(
    buf: *const c_void,
    count: c_int,
    datatype: c::Datatype,
    dest: c_int,
    tag: c_int,
    comm: c::Comm,
    request: *mut c::Request,
) -> c::ReturnStatus {
    *request = isend(buf, count, datatype, dest, tag, comm);
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Irecv(
    buf: *mut c_void,
    count: c_int,
    datatype: c::Datatype,
    source: c_int,
    tag: c_int,
    comm: c::Comm,
    request: *mut c::Request,
) -> c::ReturnStatus {
    *request = irecv(buf, count, datatype, source, tag, comm);
    consts::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn MPI_Waitall(
    count: c_int,
    array_of_requests: *mut c::Request,
    _array_of_statuses: *mut c::Status,
) -> c::ReturnStatus {
    with_context(move |ctx, _cctx| {
        let count: isize = count.try_into().unwrap();
        let mut reqs = vec![];
        for i in 0..count {
            reqs.push((*array_of_requests.offset(i)).try_into().unwrap());
        }
        let _ = ctx.waitall(&reqs);
        consts::SUCCESS
    })
}
