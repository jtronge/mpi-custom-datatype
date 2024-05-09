use mpicd::datatype::{DatatypeResult, Buffer, PackMethod, PackContext, PackState, UnpackState};

pub struct ComplexVec(pub Vec<Vec<i32>>);

impl ComplexVec {
    /// Manually pack the complex vec.
    pub fn pack(&self) -> Vec<i32> {
        self.0
            .iter()
            .flatten()
            .map(|i| *i)
            .collect()
    }

    /// Manually unpack from a provided buffer.
    pub fn unpack_from(&mut self, packed: &[i32]) {
        let mut pos = 0;
        for row in &mut self.0 {
            let len = row.len();
            row.copy_from_slice(&packed[pos..pos + len]);
            pos += len;
        }
    }
}


impl Buffer for &ComplexVec {
    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        None
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Custom(Box::new(Context {}))
    }
}

impl Buffer for &mut ComplexVec {
    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.0.as_mut_ptr() as *mut _)
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    fn pack_method(&self) -> PackMethod {
        PackMethod::Custom(Box::new(Context {}))
    }
}

struct Context {}

impl PackContext for Context {
    unsafe fn pack_state(&mut self, src: *const u8, count: usize) -> DatatypeResult<Box<dyn PackState>> {
        Ok(Box::new(State::new(src as *mut _, count)))
    }

    unsafe fn unpack_state(&mut self, dst: *mut u8, count: usize) -> DatatypeResult<Box<dyn UnpackState>> {
        Ok(Box::new(State::new(dst as *mut _, count)))
    }

    unsafe fn packed_size(&mut self, buf: *const u8, count: usize) -> DatatypeResult<usize> {
        let v = buf as *const Vec<i32>;
        let mut size = 0;
        for i in 0..count {
            let ptr = v.offset(i as isize);
            size += (*ptr).len() * std::mem::size_of::<i32>();
        }
        Ok(size)
    }
}

pub struct State {
    /// Pointer to the buffer.
    ptr: *mut Vec<i32>,

    /// Number of vectors.
    count: usize,

    /// Next expected index into outer vector.
    next_i: usize,

    /// Next expected index into inner vector.
    next_j: usize,

    /// Next expected offset.
    next_offset: usize,
}

impl State {
    fn new(ptr: *mut Vec<i32>, count: usize) -> State {
        State {
            ptr,
            count,
            next_i: 0,
            next_j: 0,
            next_offset: 0,
        }
    }
}

impl PackState for State {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        // Assume that we are packing in order for now.
        assert_eq!(self.next_offset, offset);
        let mut elements_left = dst_size / std::mem::size_of::<i32>();
        let mut used = 0;
        while elements_left > 0 && self.next_i < self.count {
            let v = self.ptr.offset(self.next_i as isize);
            let remain = (*v).len() - self.next_j;
            let copy_count = if remain > elements_left { elements_left } else { remain };
            std::ptr::copy_nonoverlapping(
                (*v).as_ptr().offset(self.next_j as isize) as *const u8,
                dst.offset(used as isize),
                copy_count * std::mem::size_of::<i32>(),
            );
            elements_left -= copy_count;
            used += copy_count * std::mem::size_of::<i32>();
            if (self.next_j + copy_count) < (*v).len() {
                self.next_j += copy_count;
            } else {
                self.next_i += 1;
                self.next_j = 0;
            }
        }
        self.next_offset += used;
        Ok(used)
    }
}

impl UnpackState for State {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        // Assume that we are packing in order for now.
        assert_eq!(self.next_offset, offset);
        assert_eq!(src_size % std::mem::size_of::<i32>(), 0);
        let mut elements_left = src_size / std::mem::size_of::<i32>();
        let mut used = 0;
        while elements_left > 0 && self.next_i < self.count {
            let v = self.ptr.offset(self.next_i as isize);
            let remain = (*v).len() - self.next_j;
            let copy_count = if remain > elements_left { elements_left } else { remain };
            std::ptr::copy_nonoverlapping(
                src.offset(used as isize),
                (*v).as_mut_ptr().offset(self.next_j as isize) as *mut u8,
                copy_count * std::mem::size_of::<i32>(),
            );
            elements_left -= copy_count;
            used += copy_count * std::mem::size_of::<i32>();
            if (self.next_j + copy_count) < (*v).len() {
                self.next_j += copy_count;
            } else {
                self.next_i += 1;
                self.next_j = 0;
            }
        }
        self.next_offset += used;
        Ok(())
    }
}
