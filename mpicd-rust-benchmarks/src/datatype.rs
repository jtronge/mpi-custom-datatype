//! Datatypes used for benchmarking.
use mpi::traits::*;
use mpicd::datatype::{DatatypeResult, MessageCount, MessagePointer, MessageBuffer, PackedSize, PackMethod, UnpackMethod};
use crate::random::Random;

/// Benchmark datatype buffer holder.
pub enum BenchmarkDatatypeBuffer {
    /// ComplexVec type.
    DoubleVec(Option<Vec<ComplexVec>>),

    /// StructVec type.
    StructVec(Option<Vec<StructVecArray>>),
}

/// Datatype buffer for rsmpi benchmarks.
pub enum RsmpiDatatypeBuffer {
    /// Bytes type.
    Bytes(Option<Vec<Vec<u8>>>),

    /// StructVec type.
    StructVec(Option<Vec<Vec<StructVec>>>)
}

/// Latency benchmark buffer holder.
pub enum LatencyBenchmarkBuffer {
    /// ComplexVec type.
    DoubleVec(Option<(ComplexVec, ComplexVec)>),

    /// StructVec type.
    StructVec(Option<(StructVecArray, StructVecArray)>),
}

/// Latency benchmark buffer holder.
pub enum RsmpiLatencyBenchmarkBuffer {
    /// Plain byte buffers.
    Bytes(Option<(Vec<u8>, Vec<u8>)>),

    /// StructVec type.
    StructVec(Option<(Vec<StructVec>, Vec<StructVec>)>),
}

/// Trait for implementing manual unpack.
pub trait ManualPack {
    /// Unpack data from a packed buffer.
    fn manual_unpack(&mut self, data: &[u8]);

    /// Pack data into a byte buffer and return it.
    fn manual_pack(&self) -> Vec<u8>;
}

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

impl ManualPack for ComplexVec {
    /// Manually pack the complex vec.
    fn manual_pack(&self) -> Vec<u8> {
        self.0
            .iter()
            .flatten()
            .map(|i| i32::to_be_bytes(*i))
            .flatten()
            .collect()
    }

    /// Manually unpack from a provided buffer.
    fn manual_unpack(&mut self, data: &[u8]) {
        let mut pos = 0;
        for row in &mut self.0 {
            for val in row {
                *val = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
            }
        }
    }
}

impl MessagePointer for ComplexVec {
    fn ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.0.as_mut_ptr() as *mut _
    }
}

impl MessageCount for ComplexVec {
    fn count(&self) -> usize {
        self.0.len()
    }
}

impl MessageBuffer for ComplexVec {
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(State::new(self.0.as_ptr() as *mut _, self.0.len()))))
    }

    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
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

impl PackedSize for State {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok(0)
    }
}

impl PackMethod for State {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        Ok(0)
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>> {
        let outer_vec = self.ptr;
        // TODO: Note wrong size here....
        Ok(
            (0..self.count)
                .map(|i| {
                    let v = outer_vec.offset(i as isize);
                    ((*v).as_ptr() as *const _, (*v).len() * std::mem::size_of::<i32>())
                })
                .collect()
        )
    }
}

impl UnpackMethod for State {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        Ok(())
    }

    unsafe fn memory_regions(&mut self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        let outer_vec = self.ptr;
        Ok(
            (0..self.count)
                .map(|i| {
                    let v = outer_vec.offset(i as isize);
                    ((*v).as_mut_ptr() as *mut _, (*v).len() * std::mem::size_of::<i32>())
                })
                .collect()
        )
    }
}

/// Number of elements in data array.
pub const STRUCT_VEC_DATA_COUNT: usize = 2048;

/// Packed size for custom API.
pub const STRUCT_VEC_PACKED_SIZE: usize = 3 * std::mem::size_of::<i32>() + std::mem::size_of::<f64>();

/// Packed size for manual packing.
pub const STRUCT_VEC_PACKED_SIZE_TOTAL: usize = 3 * std::mem::size_of::<i32>()
                                                 + std::mem::size_of::<f64>()
                                                 + STRUCT_VEC_DATA_COUNT * std::mem::size_of::<i32>();

#[derive(Equivalence)]
pub struct StructVec {
    a: i32,
    b: i32,
    c: i32,
    d: f64,
    data: [i32; STRUCT_VEC_DATA_COUNT],
}

impl StructVec {
    pub fn new() -> StructVec {
        StructVec {
            a: 34,
            b: -2332,
            c: 2293,
            d: 1.9,
            data: [123; STRUCT_VEC_DATA_COUNT],
        }
    }
}

pub struct StructVecArray(pub Vec<StructVec>);

impl StructVecArray {
    pub fn new(size: usize) -> StructVecArray {
        assert_eq!(size % STRUCT_VEC_PACKED_SIZE_TOTAL, 0);
        assert!(size >= STRUCT_VEC_PACKED_SIZE_TOTAL);
        let count = size / STRUCT_VEC_PACKED_SIZE_TOTAL;
        StructVecArray((0..count).map(|_| StructVec::new()).collect())
    }

    pub fn update(&mut self, size: usize) {
        assert_eq!(size % STRUCT_VEC_PACKED_SIZE_TOTAL, 0);
        assert!(size >= STRUCT_VEC_PACKED_SIZE_TOTAL);
        let count = size / STRUCT_VEC_PACKED_SIZE_TOTAL;
        assert!(self.0.len() < count);
        let new = count - self.0.len();
        for _ in 0..new {
            self.0.push(StructVec::new());
        }
    }
}

impl MessageCount for StructVecArray {
    fn count(&self) -> usize {
        self.0.len()
    }
}

impl MessagePointer for StructVecArray {
    fn ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.0.as_mut_ptr() as *mut _
    }
}

impl MessageBuffer for StructVecArray {
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(StructVecState {
            data: (&self.0 as *const Vec<StructVec>) as *mut _,
        })))
    }

    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
        Some(Ok(Box::new(StructVecState {
            data: &mut self.0,
        })))
    }
}

impl ManualPack for StructVecArray {
    fn manual_unpack(&mut self, data: &[u8]) {
        assert_eq!(data.len() % STRUCT_VEC_PACKED_SIZE_TOTAL, 0);
        assert_eq!(data.len() / STRUCT_VEC_PACKED_SIZE_TOTAL, self.0.len());
        let mut pos = 0;
        let array_len = STRUCT_VEC_DATA_COUNT * std::mem::size_of::<i32>();
        for elem in &mut self.0 {
            elem.a = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.b = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.c = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.d = f64::from_be_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().offset(pos as isize),
                    elem.data.as_mut_ptr() as *mut u8,
                    array_len,
                );
            }
            pos += array_len;
        }
    }

    fn manual_pack(&self) -> Vec<u8> {
        let mut data = vec![0; self.0.len() * STRUCT_VEC_PACKED_SIZE_TOTAL];
        let mut pos = 0;
        let array_len = STRUCT_VEC_DATA_COUNT * std::mem::size_of::<i32>();
        for elem in &self.0 {
            &data[pos..pos+4].copy_from_slice(&elem.a.to_be_bytes());
            pos += 4;
            &data[pos..pos+4].copy_from_slice(&elem.b.to_be_bytes());
            pos += 4;
            &data[pos..pos+4].copy_from_slice(&elem.c.to_be_bytes());
            pos += 4;
            &data[pos..pos+8].copy_from_slice(&elem.d.to_be_bytes());
            pos += 8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    elem.data.as_ptr() as *const u8,
                    data.as_mut_ptr().offset(pos as isize),
                    array_len,
                );
            }
            pos += array_len;
        }
        data
    }
}

pub struct StructVecState {
    data: *mut Vec<StructVec>,
}

impl PackedSize for StructVecState {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok((*self.data).len() * STRUCT_VEC_PACKED_SIZE)
    }
}

impl PackMethod for StructVecState {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        // Assume we get a full non-fragmented buffer for now.
        let used = (*self.data).len() * STRUCT_VEC_PACKED_SIZE;
        assert_eq!(dst_size, used);
        let mut pos = 0;
        for elem in &(*self.data) {
            std::ptr::copy_nonoverlapping(elem.a.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.b.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.c.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.d.to_be_bytes().as_ptr(), dst.offset(pos), 8);
            pos += 8;
        }
        Ok(used)
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>> {
        Ok(
            (*self.data)
                .iter()
                .map(|elem| {
                    let len = elem.data.len() * std::mem::size_of::<i32>();
                    (elem.data.as_ptr() as *const u8, len)
                })
                .collect()
        )
    }
}

impl UnpackMethod for StructVecState {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        // Assume we get a full non-fragmented buffer for now.
        let used = (*self.data).len() * STRUCT_VEC_PACKED_SIZE;
        let mut tmp = [0; 8];
        assert_eq!(src_size, used);
        let mut pos = 0;
        for elem in &mut (*self.data) {
            std::ptr::copy_nonoverlapping(src.offset(pos), tmp.as_mut_ptr(), 4);
            pos += 4;
            elem.a = i32::from_be_bytes(tmp[..4].try_into().unwrap());
            std::ptr::copy_nonoverlapping(src.offset(pos), tmp.as_mut_ptr(), 4);
            pos += 4;
            elem.b = i32::from_be_bytes(tmp[..4].try_into().unwrap());
            std::ptr::copy_nonoverlapping(src.offset(pos), tmp.as_mut_ptr(), 4);
            pos += 4;
            elem.c = i32::from_be_bytes(tmp[..4].try_into().unwrap());
            std::ptr::copy_nonoverlapping(src.offset(pos), tmp.as_mut_ptr(), 8);
            pos += 8;
            elem.d = f64::from_be_bytes(tmp[..8].try_into().unwrap());
        }
        Ok(())
    }

    unsafe fn memory_regions(&mut self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        Ok(
            (*self.data)
                .iter_mut()
                .map(|elem| {
                    let len = elem.data.len() * std::mem::size_of::<i32>();
                    (elem.data.as_mut_ptr() as *mut u8, len)
                })
                .collect()
        )
    }
}
