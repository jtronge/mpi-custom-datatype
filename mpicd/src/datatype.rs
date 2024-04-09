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
pub enum SendKind {
    /// A contiguous datatype, no packing required.
    Contiguous,

    /// Datatype must be packed with this packer.
    Pack(Box<dyn CustomPack>),
}

/// Immutable datatype to send as a message.
pub trait SendDatatype {
    /// Return a pointer to the underlying data.
    fn as_ptr(&self) -> *const u8;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the kind of the data (i.e. whether it's contiguous or needs to be packed).
    fn kind(&self) -> SendKind;
}

/// Kind of data being received.
pub enum RecvKind {
    /// A contiguous datatype, no packing required.
    Contiguous,

    /// Datatype must be unpacked with this type.
    Unpack(Box<dyn CustomUnpack>),
}

/// Mutable datatype to be received into.
pub trait RecvDatatype {
    /// Return a pointer to the underlying data.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Return the number of elements.
    fn count(&self) -> usize;

    /// Return the kind of the data (i.e. whether it's contiguous or needs to be unpacked).
    fn kind(&mut self) -> RecvKind;
}

/// Custom pack a message datatype into a buffer.
pub trait CustomPack {
    /// Pack the datatype, returning number of bytes used.
    fn pack(&self, offset: usize, dest: &mut [u8]) -> DatatypeResult<usize>;

    /// Return the total packed size of the datatype.
    fn packed_size(&self) -> DatatypeResult<usize>;
}

pub trait CustomUnpack {
    /// Unpack the datatype.
    fn unpack(&mut self, offset: usize, source: &[u8]) -> DatatypeResult<()>;
}

/// Send datatype wrapping the ucx type.
pub(crate) struct UCXDatatype {
    datatype: ucp_datatype_t,
}

impl UCXDatatype {
    /// Create a new UCX datatype from the send datatype.
    pub(crate) unsafe fn new_send_type<D: SendDatatype>() -> UCXDatatype {
        UCXDatatype {
            datatype: rust_ucp_dt_make_contig(1) as ucp_datatype_t,
        }
    }

    /// Create a new UCX datatype from the recv datatype.
    pub(crate) unsafe fn new_recv_type<D: RecvDatatype>() -> UCXDatatype {
        UCXDatatype {
            datatype: rust_ucp_dt_make_contig(1) as ucp_datatype_t,
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

impl SendDatatype for &Vec<u32> {
    fn as_ptr(&self) -> *const u8 {
        Vec::as_ptr(self) as *const _
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn kind(&self) -> SendKind {
        SendKind::Contiguous
    }
}

impl RecvDatatype for &mut Vec<u32> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        Vec::as_mut_ptr(self) as *mut _
    }

    fn count(&self) -> usize {
        self.len() * std::mem::size_of::<u32>()
    }

    fn kind(&mut self) -> RecvKind {
        RecvKind::Contiguous
    }
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_pack(
    context: *mut c_void,
    buffer: *const c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_pack()");
    std::ptr::null_mut()
}

/// NOTE: count is the number of datatype elements (NOT bytes).
unsafe extern "C" fn start_unpack(
    context: *mut c_void,
    buffer: *mut c_void,
    count: usize,
) -> *mut c_void {
    debug!("datatype::start_unpack()");
    std::ptr::null_mut()
}

/// Determine the packed size of the datatype.
unsafe extern "C" fn packed_size(state: *mut c_void) -> usize {
    debug!("datatype::packed_size()");
    0
}

/// NOTE: offset and max_length are in bytes.
unsafe extern "C" fn pack(
    state: *mut c_void,
    offset: usize,
    dest: *mut c_void,
    max_length: usize,
) -> usize {
    debug!("datatype::pack()");
    0
}

/// NOTE: offset and length are in bytes.
unsafe extern "C" fn unpack(
    state: *mut c_void,
    offset: usize,
    src: *const c_void,
    length: usize,
) -> ucs_status_t {
    debug!("datatype::unpack()");
    UCS_OK
}

unsafe extern "C" fn finish(state: *mut c_void) {
    debug!("datatype::finish()");
}

/// Create a new UCX datatype.
unsafe fn create_datatype() -> Result<ucp_datatype_t> {
    let ops = ucp_generic_dt_ops_t {
        start_pack: Some(start_pack),
        start_unpack: Some(start_unpack),
        packed_size: Some(packed_size),
        pack: Some(pack),
        unpack: Some(unpack),
        finish: Some(finish),
    };
    let mut datatype = MaybeUninit::<ucp_datatype_t>::uninit();
    let status = ucp_dt_create_generic(&ops, std::ptr::null_mut(), datatype.as_mut_ptr());
    if status != UCS_OK {
        return Err(Error::UCXError(status));
    }
    Ok(datatype.assume_init())
}
