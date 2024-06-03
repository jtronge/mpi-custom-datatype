#[derive(Copy, Clone, Debug)]
pub enum DatatypeError {
    PackError,
    UnpackError,
    PackedSizeError,
    StateError,
    RegionError,
}

pub type DatatypeResult<T> = std::result::Result<T, DatatypeError>;

pub trait MessageCount {
    /// Return the number of elements.
    fn count(&self) -> usize;
}

pub trait MessagePointer {
    /// Return a pointer to the buffer.
    fn ptr(&self) -> *const u8;

    /// Return a mutable pointer to the buffer.
    fn ptr_mut(&mut self) -> *mut u8;
}

pub trait PackedSize {
    /// Get the total packed size of the buffer.
    unsafe fn packed_size(&self) -> DatatypeResult<usize>;
}

/// Immutable datatype to send as a message.
pub trait MessageBuffer: MessageCount + MessagePointer {
    /// Return the pack method for the data (i.e. whether it's contiguous or needs to be packed).
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        None
    }

    /// Return the unpack method for the data.
    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
        None
    }
}

pub trait PackMethod: PackedSize {
    /// Pack the buffer.
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize>;

    /// If possible, return memory regions that can be sent directly.
    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>>;
}

pub trait UnpackMethod: PackedSize {
    /// Unpack the buffer.
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()>;

    /// If possible, return memory regions that can be received into.
    unsafe fn memory_regions(&mut self) -> DatatypeResult<Vec<(*mut u8, usize)>>;
}

macro_rules! impl_buffer_primitive {
    ($ty:ty) => {
        impl MessagePointer for [$ty] {
            fn ptr(&self) -> *const u8 {
                <[$ty]>::as_ptr(self) as *const _
            }

            fn ptr_mut(&mut self) -> *mut u8 {
                <[$ty]>::as_mut_ptr(self) as *mut _
            }
        }

        impl MessageCount for [$ty] {
            fn count(&self) -> usize {
                self.len() * std::mem::size_of::<$ty>()
            }
        }

        impl MessageBuffer for [$ty] {}
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
