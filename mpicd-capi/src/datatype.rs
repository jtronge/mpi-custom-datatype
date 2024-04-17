//! Datatype management code.
use std::ffi::{c_int, c_void};
use crate::c;
use mpicd::datatype::{SendBuffer, PackMethod, RecvBuffer, UnpackMethod, PackContext, UnpackContext, Pack, Unpack, DatatypeResult, DatatypeError};
use crate::{consts, with_context};
use log::debug;

/// Custom datatype handling functions for pack code.
#[derive(Copy, Clone)]
pub(crate) struct PackInfo {
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    queryfn: c::QueryFn,
    packed_elem_size: usize,
}

/// Custom datatype handling functions for iovec code.
#[derive(Copy, Clone)]
pub(crate) struct IovecInfo {
    regfn: c::RegFn,
    reg_count: usize,
}

/// List of conversion functions to be used for a custom datatype.
#[derive(Copy, Clone)]
pub(crate) enum CustomDatatype {
    /// The datatype is to be packed.
    Pack(PackInfo),

    /// The datatype can be sent as an iovec.
    Iovec(IovecInfo),
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
        match self.custom_datatype {
            CustomDatatype::Pack(pack_info) => {
                PackMethod::Pack(Box::new(CustomBufferPacker {
                    ptr: self.ptr,
                    len: self.len,
                    pack_info,
                    resume: std::ptr::null_mut(),
                }))
            }
            CustomDatatype::Iovec(iovec_info) => panic!("iovec not supported yet"),
        }
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
        match self.custom_datatype {
            CustomDatatype::Pack(pack_info) => {
                UnpackMethod::Unpack(Box::new(CustomBufferUnpacker {
                    ptr: self.ptr_mut,
                    len: self.len,
                    pack_info,
                    resume: std::ptr::null_mut(),
                }))
            }
            CustomDatatype::Iovec(iovec_info) => panic!("iovec not supported yet"),
        }
    }
}

struct CustomBufferPacker {
    /// Data pointer.
    ptr: *const u8,

    /// Length of the input buffer.
    len: usize,

    /// Packing functions and other metadata.
    pack_info: PackInfo,

    /// Resume pointer for packing method.
    resume: *mut c_void,
}

impl PackContext for CustomBufferPacker {
    fn packer(&mut self) -> Box<dyn Pack> {
        Box::new(CustomBufferPacker {
            ptr: self.ptr,
            len: self.len,
            pack_info: self.pack_info,
            resume: std::ptr::null_mut(),
        })
    }
}

fn query_packed_size(queryfn: c::QueryFn, ptr: *const c_void, len: usize) -> DatatypeResult<usize> {
    let mut packed_size = 0;
    let func = queryfn.expect("query function is missing");
    let ret = func(
        ptr,
        len,
        &mut packed_size,
    );

    if ret == 0 {
        Ok(packed_size)
    } else {
        Err(DatatypeError::PackError)
    }
}

impl Pack for CustomBufferPacker {
    /// Pack the datatype by using the externally provided C functions.
    fn pack(&mut self, offset: usize, dest: &mut [u8]) -> DatatypeResult<()> {
        debug!("calling c packfn");

        // Pack the data.
        let ret = unsafe {
            assert!(offset < self.len);
            let ptr = self.ptr.offset(offset as isize);
            let func = self.pack_info.packfn.expect("pack function is missing");
            func(
                self.len - offset,
                ptr as *const _,
                dest.len(),
                dest.as_mut_ptr() as *mut _,
                &mut self.resume,
            )
        };

        if ret == 0 {
            Ok(())
        } else {
            Err(DatatypeError::PackError)
        }
    }

    /// Return the packed size using the externally provided C functions.
    fn packed_size(&self) -> DatatypeResult<usize> {
        unsafe {
            query_packed_size(self.pack_info.queryfn, self.ptr as *const _, self.len)
        }
    }
}

struct CustomBufferUnpacker {
    ptr: *mut u8,
    len: usize,
    pack_info: PackInfo,
    resume: *mut c_void,
}

impl UnpackContext for CustomBufferUnpacker {
    fn unpacker(&mut self) -> Box<dyn Unpack> {
        Box::new(CustomBufferUnpacker {
            ptr: self.ptr,
            len: self.len,
            pack_info: self.pack_info,
            resume: std::ptr::null_mut(),
        })
    }
}

impl Unpack for CustomBufferUnpacker {
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()> {
        debug!("calling c unpackfn");
        unsafe {
            assert!(offset < self.len);
            let ptr = self.ptr.offset(offset as isize);
            let func = self.pack_info.unpackfn.expect("query function is missing");
            let ret = func(
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
            query_packed_size(self.pack_info.queryfn, self.ptr as *const _, self.len)
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
    packed_elem_size: usize,
    regfn: c::RegFn,
    reg_count: c::Count,
    datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    with_context(move |_, cctx| {
        let custom_datatype = if regfn.is_some() {
            CustomDatatype::Iovec(IovecInfo {
                regfn,
                reg_count,
            })
        } else {
            CustomDatatype::Pack(PackInfo {
                packfn,
                unpackfn,
                queryfn,
                packed_elem_size,
            })
        };
        *datatype = cctx.add_custom_datatype(custom_datatype);
        consts::SUCCESS
    })
}
