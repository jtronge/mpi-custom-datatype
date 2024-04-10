use crate::{Result, Error};
use log::debug;
use mpicd_ucx_sys::{
    rust_ucp_dt_make_contig, ucp_datatype_t, ucp_dt_create_generic, ucp_generic_dt_ops_t, ucs_status_t, UCS_OK,
};
use std::ffi::c_void;
use std::mem::MaybeUninit;

#[derive(Copy, Clone, Debug)]
pub enum DatatypeError {
    PackError,
    UnpackError,
}

pub type DatatypeResult<T> = std::result::Result<T, DatatypeError>;

/// Kind of data being sent.
pub enum PackMethod {
    /// A contiguous datatype, no packing required.
    Contiguous,

    /// Datatype must be packed with this packer.
    Pack(Box<dyn CustomPack>),
}

/// Immutable datatype to send as a message.
pub trait SendBuffer {
    /// Return a pointer to the underlying data.
    fn as_ptr(&self) -> *const u8;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the pack moethd for the data (i.e. whether it's contiguous or needs to be packed).
    fn pack_method(&self) -> PackMethod;
}

/// Kind of data being received.
pub enum UnpackMethod {
    /// A contiguous datatype, no unpacking required.
    Contiguous,

    /// Datatype must be unpacked with this type.
    Unpack(Box<dyn CustomUnpack>),
}

/// Mutable datatype to be received into.
pub trait RecvBuffer {
    /// Return a pointer to the underlying data.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the kind of the data (i.e. whether it's contiguous or needs to be unpacked).
    fn unpack_method(&mut self) -> UnpackMethod;
}

/// Custom pack a message datatype into a buffer.
pub trait CustomPack {
    /// Pack the datatype, returning number of bytes used.
    fn pack(&mut self, offset: usize, dest: &mut [u8]) -> DatatypeResult<usize>;

    /// Return the total packed size of the datatype.
    fn packed_size(&self) -> DatatypeResult<usize>;
}

/// Custom unpack a message into a buffer.
pub trait CustomUnpack {
    /// Unpack the datatype.
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()>;

    /// Return the total packed size of the datatype.
    fn packed_size(&self) -> DatatypeResult<usize>;
}

/// Send datatype wrapping the ucx type.
pub(crate) struct UCXDatatype {
    datatype: ucp_datatype_t,
}

impl UCXDatatype {
    /// Create a new UCX datatype from the send datatype.
    pub(crate) unsafe fn new_send_type<B: SendBuffer>(data: &B) -> UCXDatatype {
        let datatype = match data.pack_method() {
            PackMethod::Contiguous => rust_ucp_dt_make_contig(1) as ucp_datatype_t,
            PackMethod::Pack(custom_pack) => {
                let pack_context = PackContext {
                    pack: Some(custom_pack),
                    unpack: None,
                };
                let pack_context = Box::into_raw(Box::new(pack_context));
                create_generic_datatype(pack_context)
                    .expect("failed to create generic datatype")
            }
        };

        UCXDatatype {
            datatype,
        }
    }

    /// Create a new UCX datatype from the recv datatype.
    pub(crate) unsafe fn new_recv_type<B: RecvBuffer>(data: &mut B) -> UCXDatatype {
        let datatype = match data.unpack_method() {
            UnpackMethod::Contiguous => rust_ucp_dt_make_contig(1) as ucp_datatype_t,
            UnpackMethod::Unpack(custom_unpack) => {
                let pack_context = PackContext {
                    pack: None,
                    unpack: Some(custom_unpack),
                };
                let pack_context = Box::into_raw(Box::new(pack_context));
                create_generic_datatype(pack_context)
                    .expect("failed to create generic datatype")
            }
        };

        UCXDatatype {
            datatype,
        }
    }

    /// Return the datatype identifier.
    pub(crate) fn dt_id(&self) -> ucp_datatype_t {
        self.datatype
    }
}

impl Drop for UCXDatatype {
    fn drop(&mut self) {
        // TODO: Free datatype if necessary.
    }
}

impl SendBuffer for &Vec<u32> {
    fn as_ptr(&self) -> *const u8 {
        Vec::as_ptr(self) as *const _
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Contiguous
    }
}

impl RecvBuffer for &mut Vec<u32> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        Vec::as_mut_ptr(self) as *mut _
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn unpack_method(&mut self) -> UnpackMethod {
        UnpackMethod::Contiguous
    }
}

/// Saved pack context.
struct PackContext {
    pack: Option<Box<dyn CustomPack>>,
    unpack: Option<Box<dyn CustomUnpack>>,
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_pack(
    context: *mut c_void,
    buffer: *const c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_pack()");

    context
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_unpack(
    context: *mut c_void,
    buffer: *mut c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_unpack()");

    context
}

/// Determine the packed size of the datatype.
unsafe extern "C" fn packed_size(state: *mut c_void) -> usize {
    debug!("datatype::packed_size()");

    let state = state as *mut PackContext;
    let size = if let Some(pack) = (*state).pack.as_ref() {
        pack
            .packed_size()
            .expect("failed to get packed size of buffer")
    } else if let Some(unpack) = (*state).unpack.as_ref() {
        unpack
            .packed_size()
            .expect("failed to get packed size of buffer")
    } else {
        panic!("Missing pack and unpack implementations for querying packed_size");
    };
    debug!("packed_size: {}", size);
    size
}

/// NOTE: offset and max_length are in bytes.
unsafe extern "C" fn pack(
    state: *mut c_void,
    offset: usize,
    dest: *mut c_void,
    max_length: usize,
) -> usize {
    debug!("datatype::pack()");

    let state = state as *mut PackContext;
    let dest = dest as *mut u8;
    let dest_slice = std::slice::from_raw_parts_mut(dest, max_length);
    (*state)
        .pack
        .as_mut()
        .expect("missing pack object")
        .pack(offset, dest_slice)
        .expect("failed to pack buffer")
}

/// NOTE: offset and length are in bytes.
unsafe extern "C" fn unpack(
    state: *mut c_void,
    offset: usize,
    src: *const c_void,
    length: usize,
) -> ucs_status_t {
    debug!("datatype::unpack(state=., offset={}, src=., length={})", offset, length);

    let state = state as *mut PackContext;
    let src = src as *mut u8;
    let src_slice = std::slice::from_raw_parts(src, length);
    (*state)
        .unpack
        .as_mut()
        .expect("missing unpack object")
        .unpack(offset, src_slice)
        .expect("failed to unpack buffer");
    UCS_OK
}

unsafe extern "C" fn finish(state: *mut c_void) {
    debug!("datatype::finish()");

    let state = state as *mut PackContext;
    let _ = Box::from_raw(state);
}

/// Create a new UCX datatype.
unsafe fn create_generic_datatype(context: *mut PackContext) -> Result<ucp_datatype_t> {
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
