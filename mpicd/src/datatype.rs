use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::rc::Rc;
use log::debug;
use mpicd_ucx_sys::{
    rust_ucp_dt_make_contig, ucp_datatype_t, ucp_dt_create_generic, ucp_generic_dt_ops_t, ucs_status_t, UCS_OK,
};
use crate::{Result, Error};

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
    Pack(Box<dyn PackContext>),
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
    Unpack(Box<dyn UnpackContext>),
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

/// Pack context for creating Pack objects.
pub trait PackContext {
    /// Create a new Pack object.
    fn packer(&mut self) -> Box<dyn Pack>;
}

/// Unpack context for creating Unpack objects.
pub trait UnpackContext {
    /// Create a new Unpack object.
    fn unpacker(&mut self) -> Box<dyn Unpack>;
}

/// Custom pack a message datatype into a buffer.
pub trait Pack {
    /// Pack the datatype.
    fn pack(&mut self, offset: usize, dest: &mut [u8]) -> DatatypeResult<()>;

    /// Return the total packed size of the datatype.
    fn packed_size(&self) -> DatatypeResult<usize>;
}

/// Custom unpack a message into a buffer.
pub trait Unpack {
    /// Unpack the datatype.
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()>;

    /// Return the total packed size of the datatype.
    fn packed_size(&self) -> DatatypeResult<usize>;
}

/// Send datatype wrapping the ucx type.
pub(crate) struct UCXDatatype {
    /// Underlying ucx datatype.
    datatype: ucp_datatype_t,

    /// Pointer to a pack context impl that will be used if this is a ucx generic datatype.
    pack_context: Option<*mut PackContextType>,
}

impl UCXDatatype {
    /// Create a new UCX datatype from the send datatype.
    pub(crate) unsafe fn new_send_type<B: SendBuffer>(data: &B) -> UCXDatatype {
        let mut pack_context: Option<*mut PackContextType> = None;

        let datatype = match data.pack_method() {
            PackMethod::Contiguous => rust_ucp_dt_make_contig(1) as ucp_datatype_t,
            PackMethod::Pack(pack_ctx) => {
                let ctx = PackContextType::Pack(pack_ctx);
                let ctx_ptr = Box::into_raw(Box::new(ctx)) as *mut _;
                let _ = pack_context.insert(ctx_ptr);
                create_generic_datatype(ctx_ptr)
                    .expect("failed to create generic datatype")
            }
        };

        UCXDatatype {
            datatype,
            pack_context,
        }
    }

    /// Create a new UCX datatype from the recv datatype.
    pub(crate) unsafe fn new_recv_type<B: RecvBuffer>(data: &mut B) -> UCXDatatype {
        let mut pack_context: Option<*mut PackContextType> = None;

        let datatype = match data.unpack_method() {
            UnpackMethod::Contiguous => rust_ucp_dt_make_contig(1) as ucp_datatype_t,
            UnpackMethod::Unpack(unpack_ctx) => {
                let ctx = PackContextType::Unpack(unpack_ctx);
                // The context_ptr is free'd in the drop method for UCXDatatype.
                let ctx_ptr = Box::into_raw(Box::new(ctx));
                pack_context.insert(ctx_ptr);
                create_generic_datatype(ctx_ptr)
                    .expect("failed to create generic datatype")
            }
        };

        UCXDatatype {
            datatype,
            pack_context,
        }
    }

    /// Return the datatype identifier.
    pub(crate) fn dt_id(&self) -> ucp_datatype_t {
        self.datatype
    }
}

impl Drop for UCXDatatype {
    fn drop(&mut self) {
        unsafe {
            if let Some(pack_context) = self.pack_context {
                let _ = Rc::from_raw(pack_context);
            }
        }
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
enum PackContextType {
    Pack(Box<dyn PackContext>),
    Unpack(Box<dyn UnpackContext>),
}

enum PackState {
    Pack(Box<dyn Pack>),
    Unpack(Box<dyn Unpack>),
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_pack(
    context: *mut c_void,
    buffer: *const c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_pack()");

    let context = context as *mut PackContextType;
    match &mut (*context) {
        PackContextType::Pack(pack_ctx) => {
            let mut packer = pack_ctx.packer();
            let packer = PackState::Pack(packer);
            Box::into_raw(Box::new(packer)) as *mut _
        }
        _ => panic!("pack context invalid"),
    }
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_unpack(
    context: *mut c_void,
    buffer: *mut c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_unpack()");

    let context = context as *mut PackContextType;
    match &mut (*context) {
        PackContextType::Unpack(unpack_ctx) => {
            let mut unpacker = unpack_ctx.unpacker();
            let unpacker = PackState::Unpack(unpacker);
            Box::into_raw(Box::new(unpacker)) as *mut _
        }
        _ => panic!("unpack context invalid"),
    }
}

/// Determine the packed size of the datatype.
unsafe extern "C" fn packed_size(state: *mut c_void) -> usize {
    debug!("datatype::packed_size()");

    let state = state as *mut PackState;
    let size = match &(*state) {
        PackState::Pack(packer) => packer.packed_size().expect("failed to get packed size of buffer"),
        PackState::Unpack(unpacker) => unpacker.packed_size().expect("failed to get packed size of buffer"),
    };
    debug!("packed_size: {}", size);
    size
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

    let state = state as *mut PackState;
    let dest = dest as *mut u8;
    let dest_slice = std::slice::from_raw_parts_mut(dest, max_length);
    match &mut (*state) {
        PackState::Pack(packer) => packer.pack(offset, dest_slice).expect("failed to pack buffer"),
        _ => panic!("Invalid pack state"),
    }

    // Must always pack max_length due to bug.
    max_length
}

/// NOTE: offset and length are in bytes.
unsafe extern "C" fn unpack(
    state: *mut c_void,
    offset: usize,
    src: *const c_void,
    length: usize,
) -> ucs_status_t {
    debug!("datatype::unpack(state=., offset={}, src={:?}, length={})", offset, src, length);

    let state = state as *mut PackState;
    let src = src as *mut u8;
    let src_slice = std::slice::from_raw_parts(src, length);
    match &mut (*state) {
        PackState::Unpack(unpacker) => unpacker.unpack(offset, src_slice).expect("failed to unpack buffer"),
        _ => panic!("Invalid unpack state"),
    }
    UCS_OK
}

unsafe extern "C" fn finish(state: *mut c_void) {
    debug!("datatype::finish()");

    let state = state as *mut PackState;
    let _ = Box::from_raw(state);
}

/// Create a new UCX datatype.
unsafe fn create_generic_datatype(context: *mut PackContextType) -> Result<ucp_datatype_t> {
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
