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
    PackedSizeError,
    StateError
}

pub type DatatypeResult<T> = std::result::Result<T, DatatypeError>;

/// Kind of data being sent.
pub enum PackMethod {
    /// A contiguous datatype, no packing required.
    Contiguous,

    /// Datatype must be packed with a custom packer.
    Custom(Box<dyn PackContext>),
}

/// Immutable datatype to send as a message.
pub trait Buffer {
    /// Return a pointer to the underlying data.
    fn as_ptr(&self) -> *const u8;

    /// Return a pointer to the underlying data.
    fn as_mut_ptr(&mut self) -> Option<*mut u8>;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the pack moethd for the data (i.e. whether it's contiguous or needs to be packed).
    fn pack_method(&self) -> PackMethod;
}

pub trait PackContext {
    /// Return a PackState pointer to start packing a buffer.
    unsafe fn pack_state(&mut self, src: *const u8, count: usize) -> DatatypeResult<Box<dyn PackState>>;

    /// Return an UnpackState pointer to start unpacking a buffer.
    unsafe fn unpack_state(&mut self, dst: *mut u8, count: usize) -> DatatypeResult<Box<dyn UnpackState>>;

    /// Return the total bytes required to pack the input buffer.
    unsafe fn packed_size(&mut self, buf: *const u8, count: usize) -> DatatypeResult<usize>;
}

pub trait PackState {
    /// Pack the buffer.
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<()>;
}

pub trait UnpackState {
    /// Unpack the buffer.
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()>;
}

/// Send datatype wrapping the ucx type.
pub(crate) struct UCXDatatype {
    /// Underlying ucx datatype.
    datatype: ucp_datatype_t,

    /// Pointer to a pack context impl that will be used if this is a ucx generic datatype.
    pack_context: Option<*mut PackContextHolder>,
}

impl UCXDatatype {
    /// Create a new UCX datatype from the send datatype.
    pub(crate) unsafe fn new_type<B: Buffer>(data: &B) -> UCXDatatype {
        let mut pack_context: Option<*mut PackContextHolder> = None;

        let datatype = match data.pack_method() {
            PackMethod::Contiguous => rust_ucp_dt_make_contig(1) as ucp_datatype_t,
            PackMethod::Custom(context) => {
                // ctx_ptr is free'd in the drop method for UCXDatatype.
                let ctx_ptr = Rc::into_raw(Rc::new(PackContextHolder {
                    context,
                })) as *mut _;
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

    /// Return the datatype identifier.
    pub(crate) fn dt_id(&self) -> ucp_datatype_t {
        self.datatype
    }
}

impl Drop for UCXDatatype {
    fn drop(&mut self) {
        unsafe {
            if let Some(pack_context) = self.pack_context {
                let _ = Box::from_raw(pack_context);
            }
        }
    }
}

impl Buffer for &Vec<u32> {
    fn as_ptr(&self) -> *const u8 {
        Vec::as_ptr(self) as *const _
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        None
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Contiguous
    }
}

impl Buffer for &mut Vec<u32> {
    fn as_ptr(&self) -> *const u8 {
        Vec::as_ptr(self) as *const _
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(Vec::as_mut_ptr(self) as *mut _)
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Contiguous
    }
}

struct PackContextHolder {
    context: Box<dyn PackContext>,
}

enum State {
    Pack(Box<dyn PackState>),
    Unpack(Box<dyn UnpackState>),
}

struct StateHolder {
    buffer: *const u8,
    count: usize,
    context: *mut PackContextHolder,
    state: State,
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_pack(
    context: *mut c_void,
    buffer: *const c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_pack()");

    let context = context as *mut PackContextHolder;
    let state = (*context).context
        .pack_state(buffer as *const _, count)
        .expect("failed to create pack state");
    Box::into_raw(Box::new(StateHolder {
        buffer: buffer as *const _,
        count,
        context,
        state: State::Pack(state),
    })) as *mut _
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_unpack(
    context: *mut c_void,
    buffer: *mut c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_unpack()");

    let context = context as *mut PackContextHolder;
    let state = (*context).context
        .unpack_state(buffer as *mut _, count)
        .expect("failed to create pack state");
    Box::into_raw(Box::new(StateHolder {
        buffer: buffer as *const _,
        count,
        context,
        state: State::Unpack(state),
    })) as *mut _
}

/// Determine the packed size of the datatype.
unsafe extern "C" fn packed_size(state: *mut c_void) -> usize {
    debug!("datatype::packed_size()");

    let state = state as *mut StateHolder;
    (*(*state).context)
        .context
        .packed_size((*state).buffer, (*state).count)
        .expect("failed to get packed size of buffer")
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
    match &mut (*state).state {
        State::Pack(pack) => pack
            .pack(offset, dest as *mut _, max_length)
            .expect("failed to pack buffer"),
        _ => panic!("invalid pack state"),
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

    let state = state as *mut StateHolder;
    match &mut (*state).state {
        State::Unpack(unpack) => unpack
            .unpack(offset, src as *const _, length)
            .expect("failed to unpack buffer"),
        _ => panic!("invalid unpack state"),
    }

    UCS_OK
}

unsafe extern "C" fn finish(state: *mut c_void) {
    debug!("datatype::finish()");

    let state = state as *mut StateHolder;
    let _ = Box::from_raw(state);
}

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
