//! Datatypes used for benchmarking.
use mpicd::datatype::{DatatypeResult, MessageBuffer, PackMethod};
use crate::random::Random;

pub struct ComplexVec(pub Vec<Vec<i32>>);

/// Fill the complex vec with randomly sized vectors summing up to count.
fn fill_complex_vec(data: &mut Vec<Vec<i32>>, mut rand: Random, count: usize) {
    let mut total = 0;
    while total < count {
        let len = rand.value() % count;
        let len = if (total + len) > count {
            count - total
        } else {
            len
        };
        let inner_data = (0..len).map(|i| (total + i) as i32).collect();
        data.push(inner_data);
        total += len;
    }
}

impl ComplexVec {
    /// Create a new complex vector from a count, random seed, and rank.
    pub fn new(count: usize, seed: usize) -> ComplexVec {
        let rand = Random::new(seed);
        let mut data = vec![];
        fill_complex_vec(&mut data, rand, count);
        ComplexVec(data)
    }

    /// Create a complex vector with just a single vec.
    pub fn single_vec(count: usize) -> ComplexVec {
        ComplexVec(vec![vec![0; count]])
    }

    /// Update an existing buffer, optimizing to avoid reallocating each time.
    pub fn update(&mut self, count: usize, seed: usize) {
        let rand = Random::new(seed);

        if count < 2048 {
            self.0.clear();
            fill_complex_vec(&mut self.0, rand, count);
        } else {
            let prev_count: usize = self.0.iter().map(|v| v.len()).sum();
            assert!(count > prev_count);
            let new_count = count - prev_count;
            fill_complex_vec(&mut self.0, rand, new_count);
        }
    }

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

impl MessageBuffer for &ComplexVec {
    fn ptr(&self) -> *mut u8 {
        self.0.as_ptr() as *mut _
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    unsafe fn pack(&mut self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(State::new(self.0.as_ptr() as *mut _, self.0.len()))))
    }
}

impl MessageBuffer for &mut ComplexVec {
    fn ptr(&self) -> *mut u8 {
        self.0.as_ptr() as *mut _
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    unsafe fn pack(&mut self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(State::new(self.0.as_ptr() as *mut _, self.0.len()))))
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

impl PackMethod for State {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
/*
        let mut size = 0;
        for i in 0..self.count {
            let ptr = self.ptr.offset(i as isize);
            size += (*ptr).len() * std::mem::size_of::<i32>();
        }
        Ok(size)
*/
        Ok(0)
    }

    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
/*
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
*/
        Ok(0)
    }

    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
/*
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
*/
        Ok(())
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        let outer_vec = self.ptr;
        Ok(
            (0..self.count)
                .map(|i| {
                    let v = outer_vec.offset(i as isize);
                    ((*v).as_mut_ptr() as *mut _, (*v).len())
                })
                .collect()
        )
    }
}
