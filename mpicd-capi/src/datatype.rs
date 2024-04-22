//! Datatype management code.
use std::ffi::{c_int, c_void};
use crate::c;
use mpicd::datatype::{
    Buffer, PackMethod, PackContext, PackState, UnpackState, MemRegionsDatatype,
    DatatypeResult, DatatypeError,
};
use crate::{consts, with_context};
use log::debug;

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
    unsafe fn regions(&self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        // TODO: How are the returned buffers freed?
        let func = self.vtable.regionfn.expect("missing memory region function");
        Ok(vec![])
    }
}

impl PackContext for CustomDatatype {
    unsafe fn pack_state(&mut self, src: *const u8, count: usize) -> DatatypeResult<Box<dyn PackState>> {
        let func = self.vtable.pack_statefn.expect("missing pack_state() function pointer");
        let mut state: *mut c_void = std::ptr::null_mut();
        let ret = func(self.context, src as *const _, count, &mut state);

        if ret == 0 {
            Ok(Box::new(CustomPackState {
                custom_datatype: self.clone(),
                state,
            }))
        } else {
            Err(DatatypeError::StateError)
        }
    }

    unsafe fn unpack_state(&mut self, dst: *mut u8, count: usize) -> DatatypeResult<Box<dyn UnpackState>> {
        let func = self.vtable.unpack_statefn.expect("missing unpack_state() function pointer");
        let mut state: *mut c_void = std::ptr::null_mut();
        let ret = func(self.context, dst as *mut _, count, &mut state);

        if ret == 0 {
            Ok(Box::new(CustomUnpackState {
                custom_datatype: self.clone(),
                state,
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
}

impl PackState for CustomPackState {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        let func = self.custom_datatype.vtable.packfn.expect("missing pack() function");
        let mut used = 0;
        let ret = func(self.state, offset, dst as *mut _, dst_size, &mut used);
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
            let func = self.custom_datatype
                .vtable
                .pack_freefn
                .expect("missing pack_free() function");
            let ret = func(self.state);
            if ret != 0 {
                panic!("failed to free custom pack state");
            }
        }
    }
}

struct CustomUnpackState {
    custom_datatype: CustomDatatype,
    state: *mut c_void,
}

impl UnpackState for CustomUnpackState {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        let func = self.custom_datatype.vtable.unpackfn.expect("missing unpack() function");
        let ret = func(self.state, offset, src as *const _, src_size);
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
            let func = self.custom_datatype
                .vtable
                .unpack_freefn
                .expect("missing unpack_free() function");
            let ret = func(self.state);
            if ret != 0 {
                panic!("failed to free custom unpack state");
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
                regionfn,
            },
            context,
        });
        consts::SUCCESS
    })
}
