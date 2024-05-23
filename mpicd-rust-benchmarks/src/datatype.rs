//! Datatypes used for benchmarking.
use mpi::traits::*;
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
    pub fn new(total_count: usize, subvector_count: usize) -> ComplexVec {
        let mut data = vec![];
        let mut total = 0;
        while total < total_count {
            let remain = total_count - total;
            let next_size = if remain > subvector_count { subvector_count } else { remain };
            data.push((0..next_size).map(|i| i.try_into().unwrap()).collect());
            total += next_size;
        }
        ComplexVec(data)
    }

    /// Update an existing buffer, optimizing to avoid reallocating each time.
    pub fn update(&mut self, total_count: usize, subvector_count: usize) {
        if total_count < subvector_count {
            self.0.clear();
            self.0.push((0..total_count).map(|i| i.try_into().unwrap()).collect())
        } else {
            let mut total = 0;
            // Extend any existing subvectors.
            for subvec in self.0.iter_mut() {
                if subvec.len() < subvector_count && (total + subvector_count) <= total_count {
                    subvec.clear();
                    subvec.resize(subvector_count, 0);
                }
                total += subvec.len();
            }

            // Append new subvectors.
            while total < total_count {
                let remain = total_count - total;
                let next_size = if remain > subvector_count { subvector_count } else { remain };
                self.0.push((0..next_size).map(|i| i.try_into().unwrap()).collect());
                total += next_size;
            }

            assert_eq!(total, total_count);
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
        Ok(0)
    }

    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        Ok(0)
    }

    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
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

#[derive(Equivalence)]
pub struct StructWithArray {
    a: i32,
    b: i32,
    c: i32,
    d: f64,
    data: [i32; 2048],
}

pub struct StructWithArrayWrapper(Vec<StructWithArray>);

impl MessageBuffer for &StructWithArrayWrapper {
    fn ptr(&self) -> *mut u8 {
        self.0.as_ptr() as *mut _
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    unsafe fn pack(&mut self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(StructWithArrayState {})))
    }
}

impl MessageBuffer for &mut StructWithArrayWrapper {
    fn ptr(&self) -> *mut u8 {
        self.0.as_ptr() as *mut _
    }

    fn count(&self) -> usize {
        self.0.len()
    }

    unsafe fn pack(&mut self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(StructWithArrayState {})))
    }
}

pub struct StructWithArrayState {}

impl PackMethod for StructWithArrayState {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok(0)
    }

    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        Ok(0)
    }

    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        Ok(())
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        Ok(vec![])
    }
}
