//! Datatype management code.
use std::ffi::{c_int, c_void};
use crate::c;
use mpicd::datatype::{SendDatatype, SendKind, RecvDatatype, RecvKind, CustomPack, CustomUnpack, DatatypeResult, DatatypeError};

struct CustomDatatype {
    ptr_mut: *mut u8,
    ptr: *const u8,
    len: usize,
}

impl SendDatatype for CustomDatatype {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.len
    }

    fn kind(&self) -> SendKind {
        SendKind::Pack(Box::new(CustomDatatypePacker {}))
    }
}

impl RecvDatatype for CustomDatatype {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr_mut
    }

    fn count(&self) -> usize {
        self.len
    }

    fn kind(&mut self) -> RecvKind {
        RecvKind::Unpack(Box::new(CustomDatatypeUnpacker {}))
    }
}

struct CustomDatatypePacker {}

impl CustomPack for CustomDatatypePacker {
    /// Pack the datatype by using the externally provided C functions.
    fn pack(&self, offset: usize, dest: &mut [u8]) -> DatatypeResult<usize> {
        Err(DatatypeError::PackError)
    }

    /// Return the packed size using hte externally provided C functions.
    fn packed_size(&self) -> DatatypeResult<usize> {
        Err(DatatypeError::PackError)
    }
}

struct CustomDatatypeUnpacker {}

impl CustomUnpack for CustomDatatypeUnpacker {
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()> {
        Err(DatatypeError::UnpackError)
    }
}

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    _packfn: c::PackFn,
    _unpackfn: c::UnpackFn,
    _queryfn: c::QueryFn,
    _regfn: c::RegFn,
    _datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    // TODO
    0
}
