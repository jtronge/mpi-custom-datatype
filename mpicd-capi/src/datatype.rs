//! Datatype management code.
use std::ffi::c_void;
use crate::c;
use mpicd::datatype::{
    Buffer, PackMethod, PackContext, PackState, UnpackState, MemRegionsDatatype,
    DatatypeResult, DatatypeError,
};
use crate::{consts, with_context};

pub(crate) enum BufferPointer {
    Const(*const u8),
    Mut(*mut u8),
}

/// Custom buffer type for utilizing pack and unpack functions in C.
pub(crate) struct CustomBuffer {
    /// Pointer to underlying buffer.
    pub(crate) ptr: BufferPointer,

    /// Length of the buffer in bytes.
    pub(crate) len: usize,

    /// Custom datatype handling functions.
    pub(crate) custom_datatype: CustomDatatype,
}

impl Buffer for CustomBuffer {
    fn as_ptr(&self) -> *const u8 {
        match self.ptr {
            BufferPointer::Const(ptr) => ptr,
            BufferPointer::Mut(ptr) => ptr,
        }
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        match self.ptr {
            BufferPointer::Mut(ptr) => Some(ptr),
            BufferPointer::Const(_) => None,
        }
    }

    fn count(&self) -> usize {
        self.len
    }

    fn pack_method(&self) -> PackMethod {
        if self.custom_datatype.supports_iovec() {
            PackMethod::MemRegions(Box::new(self.custom_datatype))
        } else {
            PackMethod::Custom(Box::new(self.custom_datatype))
        }
    }
}

/// Send buffer to be used for MPI_BYTE types.
pub(crate) struct ByteBuffer {
    pub(crate) ptr: BufferPointer,
    pub(crate) size: usize,
}

impl Buffer for ByteBuffer {
    fn as_ptr(&self) -> *const u8 {
        match self.ptr {
            BufferPointer::Const(ptr) => ptr,
            BufferPointer::Mut(ptr) => ptr,
        }
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        match self.ptr {
            BufferPointer::Mut(ptr) => Some(ptr),
            BufferPointer::Const(_) => None,
        }
    }

    fn count(&self) -> usize {
        self.size
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Contiguous
    }
}

#[derive(Copy, Clone)]
pub(crate) struct CustomDatatypeVTable {
    pack_statefn: c::PackStateFn,
    unpack_statefn: c::UnpackStateFn,
    queryfn: c::QueryFn,
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    pack_freefn: c::PackStateFreeFn,
    unpack_freefn: c::UnpackStateFreeFn,
    region_countfn: c::RegionCountFn,
    regionfn: c::RegionFn,
}

/// Custom datatype vtable and context info.
#[derive(Copy, Clone)]
pub(crate) struct CustomDatatype {
    vtable: CustomDatatypeVTable,
    context: *mut c_void,
}

impl CustomDatatype {
    pub(crate) fn supports_iovec(&self) -> bool {
        self.vtable.regionfn.is_some()
    }
}

impl MemRegionsDatatype for CustomDatatype {
    unsafe fn regions(&self, buf: *mut u8, count: usize) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        // TODO: How are the returned buffers freed?
        let region_countfn = self.vtable.region_countfn.expect("missing memory region count function");
        let mut region_count = 0;
        let ret = region_countfn(buf as *mut _, count, &mut region_count);
        if ret != 0 {
            return Err(DatatypeError::RegionError);
        }

        let regionfn = self.vtable.regionfn.expect("missing memory region function");
        let mut reg_lens = vec![0; region_count];
        let mut reg_bases = vec![std::ptr::null_mut(); region_count];
        // For now don't do anything with the types.
        let mut types = vec![consts::BYTE; region_count];
        let ret = regionfn(
            buf as *mut _,
            count,
            region_count,
            reg_lens.as_mut_ptr(),
            reg_bases.as_mut_ptr(),
            types.as_mut_ptr(),
        );
        if ret != 0 {
            return Err(DatatypeError::RegionError);
        }

        Ok(
            reg_bases
                .iter()
                .zip(reg_lens.iter())
                .map(|(ptr, len)| (*ptr as *mut u8, *len))
                .collect()
        )
    }
}

impl PackContext for CustomDatatype {
    unsafe fn pack_state(&mut self, src: *const u8, count: usize) -> DatatypeResult<Box<dyn PackState>> {
        let mut state: *mut c_void = std::ptr::null_mut();
        let ret = if let Some(func) = self.vtable.pack_statefn {
            func(self.context, src as *const _, count, &mut state)
        } else {
            0
        };

        if ret == 0 {
            Ok(Box::new(CustomPackState {
                custom_datatype: self.clone(),
                state,
                buf: src as *const _,
                count,
            }))
        } else {
            Err(DatatypeError::StateError)
        }
    }

    unsafe fn unpack_state(&mut self, dst: *mut u8, count: usize) -> DatatypeResult<Box<dyn UnpackState>> {
        let mut state: *mut c_void = std::ptr::null_mut();
        let ret = if let Some(func) = self.vtable.unpack_statefn {
            func(self.context, dst as *mut _, count, &mut state)
        } else {
            0
        };

        if ret == 0 {
            Ok(Box::new(CustomUnpackState {
                custom_datatype: self.clone(),
                state,
                buf: dst as *mut _,
                count,
            }))
        } else {
            Err(DatatypeError::StateError)
        }
    }

    unsafe fn packed_size(&mut self, buf: *const u8, count: usize) -> DatatypeResult<usize> {
        let func = self.vtable.queryfn.expect("missing query() function pointer");
        let mut packed_size = 0;
        let ret = func(self.context, buf as *const _, count, &mut packed_size);
        if ret == 0 {
            Ok(packed_size)
        } else {
            Err(DatatypeError::PackedSizeError)
        }
    }
}

struct CustomPackState {
    custom_datatype: CustomDatatype,
    state: *mut c_void,
    buf: *const c_void,
    count: usize,
}

impl PackState for CustomPackState {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        let func = self.custom_datatype.vtable.packfn.expect("missing pack() function");
        let mut used = 0;
        let ret = func(self.state, self.buf, self.count, offset, dst as *mut _, dst_size, &mut used);
        if ret == 0 {
            Ok(used)
        } else {
            Err(DatatypeError::PackError)
        }
    }
}

impl Drop for CustomPackState {
    fn drop(&mut self) {
        unsafe {
            if let Some(func) = self.custom_datatype.vtable.pack_freefn {
                let ret = func(self.state);
                if ret != 0 {
                    panic!("failed to free custom pack state");
                }
            }
        }
    }
}

struct CustomUnpackState {
    custom_datatype: CustomDatatype,
    state: *mut c_void,
    buf: *mut c_void,
    count: usize,
}

impl UnpackState for CustomUnpackState {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        let func = self.custom_datatype.vtable.unpackfn.expect("missing unpack() function");
        let ret = func(self.state, self.buf, self.count, offset, src as *const _, src_size);
        if ret == 0 {
            Ok(())
        } else {
            Err(DatatypeError::UnpackError)
        }
    }
}

impl Drop for CustomUnpackState {
    fn drop(&mut self) {
        unsafe {
            if let Some(func) = self.custom_datatype.vtable.unpack_freefn {
                let ret = func(self.state);
                if ret != 0 {
                    panic!("failed to free custom unpack state");
                }
            }
        }
    }
}

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    pack_statefn: c::PackStateFn,
    unpack_statefn: c::UnpackStateFn,
    queryfn: c::QueryFn,
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    pack_freefn: c::PackStateFreeFn,
    unpack_freefn: c::UnpackStateFreeFn,
    region_countfn: c::RegionCountFn,
    regionfn: c::RegionFn,
    context: *mut c_void,
    datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    with_context(move |_, cctx| {
        *datatype = cctx.add_custom_datatype(CustomDatatype {
            vtable: CustomDatatypeVTable {
                pack_statefn,
                unpack_statefn,
                queryfn,
                packfn,
                unpackfn,
                pack_freefn,
                unpack_freefn,
                region_countfn,
                regionfn,
            },
            context,
        });
        consts::SUCCESS
    })
}
