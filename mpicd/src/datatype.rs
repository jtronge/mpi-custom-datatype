use std::ffi::c_void;
use std::mem::MaybeUninit;
use log::debug;
use mpicd_ucx_sys::{
    rust_ucp_dt_make_contig, rust_ucp_dt_make_iov, ucp_datatype_t, ucp_dt_create_generic, ucp_generic_dt_ops_t, ucp_dt_iov_t, ucs_status_t, UCS_OK,
};
use crate::{Result, Error};

#[derive(Copy, Clone, Debug)]
pub enum DatatypeError {
    PackError,
    UnpackError,
    PackedSizeError,
    StateError,
    RegionError,
}

pub type DatatypeResult<T> = std::result::Result<T, DatatypeError>;

/// Immutable datatype to send as a message.
pub trait MessageBuffer {
    /// Return a pointer to the underlying data.
    fn ptr(&self) -> *mut u8;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the pack moethd for the data (i.e. whether it's contiguous or needs to be packed).
    unsafe fn pack(&mut self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        None
    }
}

pub trait PackMethod {
    /// Get the total packed size of the buffer.
    unsafe fn packed_size(&self) -> DatatypeResult<usize>;

    /// Pack the buffer.
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize>;

    /// Unpack the buffer.
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()>;

    /// If possible, return memory regions that can be sent directly.
    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*mut u8, usize)>>;
}

/*
/// Send datatype wrapping the ucx type.
pub(crate) struct GenericDatatype {
    /// Underlying ucx datatype.
    datatype: ucp_datatype_t,

    /// Pointer to a pack context impl that will be used.
    pack_context: *mut PackContextHolder,
}

impl GenericDatatype {
    /// Create a new UCX datatype from the buffer.
    pub(crate) unsafe fn new<P: Pack>(context: Pack) -> GenericDatatype {
        // ctx_ptr is free'd in the drop method for UCXBuffer.
        let pack_context = Box::into_raw(Box::new(PackContextHolder {
            context,
        })) as *mut _;
        let datatype = create_generic_datatype(pack_context)
            .expect("failed to create generic datatype");

        GenericDatatype {
            datatype,
            pack_context,
        }
    }

    /// Return the datatype identifier.
    pub(crate) fn dt_id(&self) -> ucp_datatype_t {
        self.datatype
    }
}

impl Drop for GenericDatatype {
    fn drop(&mut self) {
        unsafe {
            let _ = Box::from_raw(self.pack_context);
        }
    }
}
*/

macro_rules! impl_buffer_primitive {
    ($ty:ty) => {
        impl MessageBuffer for &[$ty] {
            fn ptr(&self) -> *mut u8 {
                <[$ty]>::as_ptr(self) as *mut _
            }

            fn count(&self) -> usize {
                self.len() * std::mem::size_of::<$ty>()
            }
        }

        impl MessageBuffer for &mut [$ty] {
            fn ptr(&self) -> *mut u8 {
                <[$ty]>::as_ptr(self) as *mut _
            }

            fn count(&self) -> usize {
                self.len() * std::mem::size_of::<i32>()
            }
        }
    };
}

impl_buffer_primitive!(u8);
impl_buffer_primitive!(u16);
impl_buffer_primitive!(u32);
impl_buffer_primitive!(u64);
impl_buffer_primitive!(i8);
impl_buffer_primitive!(i16);
impl_buffer_primitive!(i32);
impl_buffer_primitive!(i64);
impl_buffer_primitive!(f32);
impl_buffer_primitive!(f64);

struct PackMethodHolder {
    packer: Box<dyn PackMethod>,
}

struct StateHolder {
    /// Buffer saved for getting the packed size later.
    buffer: *const u8,

    /// Number of elements in the buffer.
    count: usize,

    /// The pack method .
    pack_method: *mut PackMethodHolder,

    /// Packed size of the buffer.
    packed_size: Option<usize>,
}

impl StateHolder {
    /// Get the packed size for the buffer.
    unsafe fn get_packed_size(&self) -> usize {
        (*self.pack_method)
            .packer
            .packed_size()
            .expect("failed to get packed size of buffer")
    }
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_pack(
    context: *mut c_void,
    buffer: *const c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_pack()");

    let pack_method = context as *mut PackMethodHolder;
    Box::into_raw(Box::new(StateHolder {
        buffer: buffer as *const _,
        count,
        pack_method,
        packed_size: None,
    })) as *mut _
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_unpack(
    context: *mut c_void,
    buffer: *mut c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_unpack()");

    let pack_method = context as *mut PackMethodHolder;
    Box::into_raw(Box::new(StateHolder {
        buffer: buffer as *const _,
        count,
        pack_method,
        packed_size: None,
    })) as *mut _
}

/// Determine the packed size of the datatype.
unsafe extern "C" fn packed_size(state: *mut c_void) -> usize {
    debug!("datatype::packed_size()");

    let state = state as *mut StateHolder;
    (*state).get_packed_size()
}

/// Using the 'state' object, pack 'max_length' data into 'dest', returning the
/// number of bytes packed.
///
/// NOTE: This seems to have problems if the number of bytes packed are less
///       than `max_length`.
unsafe extern "C" fn pack(
    state: *mut c_void,
    offset: usize,
    dest: *mut c_void,
    max_length: usize,
) -> usize {
    debug!("datatype::pack(offset={}, dst={:?}, max_length={})", offset, dest, max_length);

    let state = state as *mut StateHolder;

    let packed_size = if let Some(packed_size) = (*state).packed_size {
        packed_size
    } else {
        let packed_size = (*state).get_packed_size();
        let _ = (*state).packed_size.insert(packed_size);
        packed_size
    };
    assert!(offset < packed_size);

    // Set the max length based on the offset and packed size.
    let max_length = if (offset + max_length) < packed_size { max_length } else { packed_size - offset };
    let used = (*(*state).pack_method)
        .packer
        .pack(offset, dest as *mut _, max_length)
        .expect("failed to pack buffer");
    assert!(used <= max_length);

    used
}

/// NOTE: offset and length are in bytes.
unsafe extern "C" fn unpack(
    state: *mut c_void,
    offset: usize,
    src: *const c_void,
    length: usize,
) -> ucs_status_t {
    debug!("datatype::unpack(state=., offset={}, src={:?}, length={})", offset, src, length);

    let state = state as *mut StateHolder;
    (*(*state).pack_method)
        .packer
        .unpack(offset, src as *const _, length)
        .expect("failed to unpack buffer");

    UCS_OK
}

unsafe extern "C" fn finish(state: *mut c_void) {
    debug!("datatype::finish()");

    let state = state as *mut StateHolder;
    let _ = Box::from_raw(state);
}

/*
/// Create a new UCX datatype.
unsafe fn create_generic_datatype(context: *mut PackContextHolder) -> Result<ucp_datatype_t> {
    let ops = ucp_generic_dt_ops_t {
        start_pack: Some(start_pack),
        start_unpack: Some(start_unpack),
        packed_size: Some(packed_size),
        pack: Some(pack),
        unpack: Some(unpack),
        finish: Some(finish),
    };
    let mut datatype = MaybeUninit::<ucp_datatype_t>::uninit();
    let status = ucp_dt_create_generic(&ops, context as *mut _, datatype.as_mut_ptr());
    if status != UCS_OK {
        return Err(Error::UCXError(status));
    }
    Ok(datatype.assume_init())
}
*/
