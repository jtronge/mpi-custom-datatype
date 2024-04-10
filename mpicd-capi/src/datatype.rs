//! Datatype management code.
use std::ffi::{c_int, c_void};
use crate::c;
use mpicd::{
    datatype::{SendBuffer, PackMethod, RecvBuffer, UnpackMethod, CustomPack, CustomUnpack, DatatypeResult, DatatypeError},
};
use crate::{consts, with_context};
use log::debug;

/// List of conversion functions to be used for a custom datatype.
#[derive(Copy, Clone)]
pub(crate) struct CustomDatatype {
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    queryfn: c::QueryFn,
}

/// Custom buffer type for utilizing pack and unpack functions in C.
pub(crate) struct CustomBuffer {
    /// Mutable pointer, if receive buffer.
    pub(crate) ptr_mut: *mut u8,

    /// Immutable pointer, if send buffer to be packed.
    pub(crate) ptr: *const u8,

    /// Length of the buffer in bytes.
    pub(crate) len: usize,

    /// Custom datatype handling functions.
    pub(crate) custom_datatype: CustomDatatype,
}

impl SendBuffer for CustomBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.len
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Pack(Box::new(CustomBufferPacker {
            ptr: self.ptr,
            len: self.len,
            custom_datatype: self.custom_datatype,
            resume: std::ptr::null_mut(),
        }))
    }
}

impl RecvBuffer for CustomBuffer {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr_mut
    }

    fn count(&self) -> usize {
        self.len
    }

    fn unpack_method(&mut self) -> UnpackMethod {
        UnpackMethod::Unpack(Box::new(CustomBufferUnpacker {
            ptr: self.ptr_mut,
            len: self.len,
            custom_datatype: self.custom_datatype,
            resume: std::ptr::null_mut(),
        }))
    }
}

struct CustomBufferPacker {
    ptr: *const u8,
    len: usize,
    custom_datatype: CustomDatatype,
    resume: *mut c_void,
}

impl CustomPack for CustomBufferPacker {
    /// Pack the datatype by using the externally provided C functions.
    fn pack(&mut self, offset: usize, dest: &mut [u8]) -> DatatypeResult<usize> {
        debug!("calling c packfn");
        unsafe {
            assert!(offset < self.len);
            let ptr = self.ptr.offset(offset as isize);
            let mut used = 0;
            let ret = (self.custom_datatype.packfn)(
                self.len - offset,
                ptr as *const _,
                dest.len(),
                dest.as_mut_ptr() as *mut _,
                &mut used,
                &mut self.resume,
            );

            if ret == 0 {
                Ok(used)
            } else {
                Err(DatatypeError::PackError)
            }
        }
    }

    /// Return the packed size using the externally provided C functions.
    fn packed_size(&self) -> DatatypeResult<usize> {
        unsafe {
            let mut packed_size = 0;
            let ret = (self.custom_datatype.queryfn)(
                self.ptr as *const _,
                self.len,
                &mut packed_size,
            );

            if ret == 0 {
                Ok(packed_size)
            } else {
                Err(DatatypeError::PackError)
            }
        }
    }
}

struct CustomBufferUnpacker {
    ptr: *mut u8,
    len: usize,
    custom_datatype: CustomDatatype,
    resume: *mut c_void,
}

impl CustomUnpack for CustomBufferUnpacker {
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()> {
        debug!("calling c unpackfn");
        unsafe {
            assert!(offset < self.len);
            let ptr = self.ptr.offset(offset as isize);
            let ret = (self.custom_datatype.unpackfn)(
                source.len(),
                source.as_ptr() as *const _,
                self.len - offset,
                ptr as *mut _,
                &mut self.resume,
            );

            if ret == 0 {
                Ok(())
            } else {
                Err(DatatypeError::UnpackError)
            }
        }
    }

    /// Return the packed size using the externally provided C functions.
    fn packed_size(&self) -> DatatypeResult<usize> {
        unsafe {
            let mut packed_size = 0;
            let ret = (self.custom_datatype.queryfn)(
                self.ptr as *const _,
                self.len,
                &mut packed_size,
            );

            if ret == 0 {
                Ok(packed_size)
            } else {
                Err(DatatypeError::PackError)
            }
        }
    }
}

/// Send buffer to be used for MPI_BYTE types.
pub(crate) struct ByteSendBuffer {
    pub(crate) ptr: *const u8,
    pub(crate) size: usize,
}

impl SendBuffer for ByteSendBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Contiguous
    }
}

/// Receive buffer to be used for MPI_BYTE types.
pub(crate) struct ByteRecvBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
}

impl RecvBuffer for ByteRecvBuffer {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn count(&self) -> usize {
        self.size
    }

    fn unpack_method(&mut self) -> UnpackMethod {
        UnpackMethod::Contiguous
    }
}

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    queryfn: c::QueryFn,
    _regfn: c::RegFn,
    _reg_count: c::Count,
    datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    with_context(move |_, cctx| {
        let custom_datatype = CustomDatatype {
            packfn,
            unpackfn,
            queryfn,
        };
        *datatype = cctx.add_custom_datatype(CustomDatatype {
            packfn,
            unpackfn,
            queryfn,
        });
        consts::SUCCESS
    })
}
