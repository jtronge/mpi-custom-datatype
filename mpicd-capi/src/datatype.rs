//! Datatype management code.
use std::ffi::{c_void, c_int};
use crate::c;
use mpicd::datatype::{DatatypeResult, DatatypeError, MessageBuffer, MessageCount, MessagePointer, PackMethod, UnpackMethod, PackedSize};
use crate::{consts, with_context};

/// Custom buffer type for utilizing pack and unpack functions in C.
pub(crate) struct CustomBuffer {
    /// Pointer to underlying buffer.
    pub(crate) ptr: *mut u8,

    /// Length of the buffer in bytes.
    pub(crate) len: usize,

    /// Custom datatype handling functions.
    pub(crate) custom_datatype: CustomDatatype,
}

impl CustomBuffer {
    unsafe fn state(&self) -> DatatypeResult<*mut c_void> {
        let mut state: *mut c_void = std::ptr::null_mut();
        let ret = if let Some(func) = self.custom_datatype.vtable.statefn {
            func(self.custom_datatype.context, self.ptr as *const _, self.len, &mut state)
        } else {
            0
        };

        if ret == 0 {
            Ok(state)
        } else {
            Err(DatatypeError::StateError)
        }
    }
}


impl MessagePointer for CustomBuffer {
    fn ptr(&self) -> *const u8 {
        self.ptr as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl MessageCount for CustomBuffer {
    fn count(&self) -> usize {
        self.len
    }
}

impl MessageBuffer for CustomBuffer {
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        let state = self.state();
        if let Err(err) = state {
            return Some(Err(err));
        }
        let state = state.expect("expected state pointer");

        Some(Ok(Box::new(CustomPackMethod {
            custom_datatype: self.custom_datatype,
            state,
            ptr: self.ptr as *const _,
            count: self.len,
        })))
    }

    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
        let state = self.state();
        if let Err(err) = state {
            return Some(Err(err));
        }
        let state = state.expect("expected state pointer");

        Some(Ok(Box::new(CustomPackMethod {
            custom_datatype: self.custom_datatype,
            state,
            ptr: self.ptr as *const _,
            count: self.len,
        })))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct CustomDatatypeVTable {
    statefn: c::StateFn,
    state_freefn: c::StateFreeFn,
    queryfn: c::QueryFn,
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    region_countfn: c::RegionCountFn,
    regionfn: c::RegionFn,
}

/// Custom datatype vtable and context info.
#[derive(Copy, Clone)]
pub(crate) struct CustomDatatype {
    vtable: CustomDatatypeVTable,
    context: *mut c_void,
}

struct CustomPackMethod {
    custom_datatype: CustomDatatype,
    state: *mut c_void,
    ptr: *const c_void,
    count: usize,
}

impl CustomPackMethod {
    unsafe fn regions(&self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        if self.custom_datatype.vtable.region_countfn.is_none() {
            return Ok(vec![]);
        }

        let region_countfn = self.custom_datatype.vtable.region_countfn.unwrap();
        let mut region_count = 0;
        let ret = region_countfn(self.state, self.ptr as *mut _, self.count, &mut region_count);
        if ret != 0 {
            return Err(DatatypeError::RegionError);
        }

        let regionfn = self.custom_datatype.vtable.regionfn.expect("missing memory region function");
        let mut reg_lens = vec![0; region_count];
        let mut reg_bases = vec![std::ptr::null_mut(); region_count];
        // Ignore the types for now.
        let mut types = vec![consts::BYTE; region_count];
        let ret = regionfn(
            self.state,
            self.ptr as *mut _,
            self.count,
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

impl PackedSize for CustomPackMethod {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        if let Some(func) = self.custom_datatype.vtable.queryfn {
            let mut packed_size = 0;
            let ret = func(self.state, self.ptr as *const _, self.count, &mut packed_size);
            if ret == 0 {
                Ok(packed_size)
            } else {
                Err(DatatypeError::PackedSizeError)
            }
        } else {
            Ok(0)
        }
    }
}

impl UnpackMethod for CustomPackMethod {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        if let Some(func) = self.custom_datatype.vtable.unpackfn {
            let ret = func(self.state, self.ptr as *mut _, self.count, offset, src as *const _, src_size);
            if ret == 0 {
                Ok(())
            } else {
                Err(DatatypeError::UnpackError)
            }
        } else {
            Ok(())
        }
    }

    unsafe fn memory_regions(&mut self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        self.regions()
    }
}

impl PackMethod for CustomPackMethod {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        if let Some(func) = self.custom_datatype.vtable.packfn {
            let mut used = 0;
            let ret = func(self.state, self.ptr, self.count, offset, dst as *mut _, dst_size, &mut used);
            if ret == 0 {
                Ok(used)
            } else {
                Err(DatatypeError::PackError)
            }
        } else {
            Ok(0)
        }
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>> {
        self.regions()
            .map(|v| v.iter().map(|(ptr, count)| (*ptr as *const _, *count)).collect())
    }
}

impl Drop for CustomPackMethod {
    fn drop(&mut self) {
        unsafe {
            if let Some(func) = self.custom_datatype.vtable.state_freefn {
                let ret = func(self.state);
                if ret != 0 {
                    panic!("failed to free custom pack state");
                }
            }
        }
    }
}

/// Send buffer to be used for MPI_BYTE types.
pub(crate) struct ByteBuffer {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
}

impl MessagePointer for ByteBuffer {
    fn ptr(&self) -> *const u8 {
        self.ptr as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl MessageCount for ByteBuffer {
    fn count(&self) -> usize {
        self.size
    }
}

impl MessageBuffer for ByteBuffer {}

/// Create a non-dynamic custom MPI_Datatype.
#[no_mangle]
pub unsafe extern "C" fn MPI_Type_create_custom(
    statefn: c::StateFn,
    state_freefn: c::StateFreeFn,
    queryfn: c::QueryFn,
    packfn: c::PackFn,
    unpackfn: c::UnpackFn,
    region_countfn: c::RegionCountFn,
    regionfn: c::RegionFn,
    context: *mut c_void,
    _inorder: c_int,
    datatype: *mut c::Datatype,
) -> c::ReturnStatus {
    with_context(move |_, cctx| {
        *datatype = cctx.add_custom_datatype(CustomDatatype {
            vtable: CustomDatatypeVTable {
                statefn,
                state_freefn,
                queryfn,
                packfn,
                unpackfn,
                region_countfn,
                regionfn,
            },
            context,
        });
        consts::SUCCESS
    })
}
