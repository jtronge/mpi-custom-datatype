//! Datatypes used for benchmarking.
use mpi::traits::*;
use mpicd::datatype::{DatatypeResult, MessageCount, MessagePointer, MessageBuffer, PackedSize, PackMethod, UnpackMethod};

/// Benchmark datatype buffer holder.
pub enum BenchmarkDatatypeBuffer {
    /// ComplexVec type.
    DoubleVec(Option<Vec<ComplexVec>>),

    /// StructVec type.
    StructVec(Option<Vec<StructVecArray>>),

    /// StructSimple type.
    StructSimple(Option<Vec<StructSimpleArray>>),

    /// StructSimpleNoGap type.
    StructSimpleNoGap(Option<Vec<StructSimpleNoGapArray>>),
}

/// Datatype buffer for rsmpi benchmarks.
pub enum RsmpiDatatypeBuffer {
    /// Bytes type.
    Bytes(Option<Vec<Vec<u8>>>),

    /// StructVec type.
    StructVec(Option<Vec<Vec<StructVec>>>),

    /// StructSimple type.
    StructSimple(Option<Vec<Vec<StructSimple>>>),

    /// StructSimpleNoGap type.
    StructSimpleNoGap(Option<Vec<Vec<StructSimpleNoGap>>>),
}

/// Latency benchmark buffer holder.
pub enum LatencyBenchmarkBuffer {
    /// ComplexVec type.
    DoubleVec(Option<(ComplexVec, ComplexVec)>),

    /// StructVec type.
    StructVec(Option<(StructVecArray, StructVecArray)>),

    /// StructSimple type.
    StructSimple(Option<(StructSimpleArray, StructSimpleArray)>),

    /// StructSimpleNoGap type.
    StructSimpleNoGap(Option<(StructSimpleNoGapArray, StructSimpleNoGapArray)>),
}

/// Latency benchmark buffer holder.
pub enum RsmpiLatencyBenchmarkBuffer {
    /// Plain byte buffers.
    Bytes(Option<(Vec<u8>, Vec<u8>)>),

    /// StructVec type.
    StructVec(Option<(Vec<StructVec>, Vec<StructVec>)>),

    /// StructSimple type.
    StructSimple(Option<(Vec<StructSimple>, Vec<StructSimple>)>),

    /// StructSimple type.
    StructSimpleNoGap(Option<(Vec<StructSimpleNoGap>, Vec<StructSimpleNoGap>)>),
}

/// Trait for implementing manual unpack.
pub trait ManualPack {
    /// Return the size of a packed buffer.
    fn packed_size(&self) -> usize;

    /// Pack data into a byte buffer and return it.
    fn manual_pack(&self) -> Vec<u8>;

    /// Unpack data from a packed buffer.
    fn manual_unpack(&mut self, data: &[u8]);
}

pub struct ComplexVec(pub Vec<Vec<i32>>);

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
    fn packed_size(&self) -> usize {
        self.0
            .iter()
            .map(|v| v.len())
            .sum()
    }

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
}

impl State {
    fn new(ptr: *mut Vec<i32>, count: usize) -> State {
        State {
            ptr,
            count,
        }
    }
}

impl PackedSize for State {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok(0)
    }
}

impl PackMethod for State {
    unsafe fn pack(&mut self, _offset: usize, _dst: *mut u8, _dst_size: usize) -> DatatypeResult<usize> {
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
    unsafe fn unpack(&mut self, _offset: usize, _src: *const u8, _src_size: usize) -> DatatypeResult<()> {
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
#[repr(C)]
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
    fn packed_size(&self) -> usize {
        self.0.len() * STRUCT_VEC_PACKED_SIZE_TOTAL
    }

    fn manual_pack(&self) -> Vec<u8> {
        let mut data = vec![0; self.0.len() * STRUCT_VEC_PACKED_SIZE_TOTAL];
        let mut pos = 0;
        let array_len = STRUCT_VEC_DATA_COUNT * std::mem::size_of::<i32>();
        for elem in &self.0 {
            let _ = &data[pos..pos+4].copy_from_slice(&elem.a.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+4].copy_from_slice(&elem.b.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+4].copy_from_slice(&elem.c.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+8].copy_from_slice(&elem.d.to_be_bytes());
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
        let mut i = offset / STRUCT_VEC_PACKED_SIZE;
        let mut pos: isize = 0;
        while i < (*self.data).len() && (pos + STRUCT_VEC_PACKED_SIZE as isize) <= dst_size as isize {
            let elem = &(*self.data)[i];
            std::ptr::copy_nonoverlapping(elem.a.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.b.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.c.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.d.to_be_bytes().as_ptr(), dst.offset(pos), 8);
            pos += 8;
            i += 1;
        }
        Ok(pos.try_into().expect("failed to convert pos from isize to usize"))
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
        assert_eq!(offset, 0);
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

/// Packed size for the simple struct.
pub const STRUCT_SIMPLE_PACKED_SIZE: usize = 3 * std::mem::size_of::<i32>() + std::mem::size_of::<f64>();

#[derive(Equivalence)]
#[repr(C)]
pub struct StructSimple {
    a: i32,
    b: i32,
    c: i32,
    d: f64,
}

impl StructSimple {
    pub fn new() -> StructSimple {
        StructSimple {
            a: 34,
            b: -2332,
            c: 2293,
            d: 1.9,
        }
    }
}

pub struct StructSimpleArray(pub Vec<StructSimple>);

impl StructSimpleArray {
    pub fn new(size: usize) -> StructSimpleArray {
        assert_eq!(size % STRUCT_SIMPLE_PACKED_SIZE, 0);
        assert!(size >= STRUCT_SIMPLE_PACKED_SIZE);
        let count = size / STRUCT_SIMPLE_PACKED_SIZE;
        StructSimpleArray((0..count).map(|_| StructSimple::new()).collect())
    }

    pub fn update(&mut self, size: usize) {
        assert_eq!(size % STRUCT_SIMPLE_PACKED_SIZE, 0);
        assert!(size >= STRUCT_SIMPLE_PACKED_SIZE);
        let count = size / STRUCT_SIMPLE_PACKED_SIZE;
        assert!(self.0.len() < count);
        let new = count - self.0.len();
        for _ in 0..new {
            self.0.push(StructSimple::new());
        }
    }
}

impl MessageCount for StructSimpleArray {
    fn count(&self) -> usize {
        self.0.len()
    }
}

impl MessagePointer for StructSimpleArray {
    fn ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.0.as_mut_ptr() as *mut _
    }
}

impl MessageBuffer for StructSimpleArray {
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(StructSimpleState {
            data: (&self.0 as *const Vec<StructSimple>) as *mut _,
        })))
    }

    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
        Some(Ok(Box::new(StructSimpleState {
            data: &mut self.0,
        })))
    }
}

impl ManualPack for StructSimpleArray {
    fn packed_size(&self) -> usize {
        self.0.len() * STRUCT_SIMPLE_PACKED_SIZE
    }

    fn manual_pack(&self) -> Vec<u8> {
        let mut data = vec![0; self.0.len() * STRUCT_SIMPLE_PACKED_SIZE];
        let mut pos = 0;
        for elem in &self.0 {
            let _ = &data[pos..pos+4].copy_from_slice(&elem.a.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+4].copy_from_slice(&elem.b.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+4].copy_from_slice(&elem.c.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+8].copy_from_slice(&elem.d.to_be_bytes());
            pos += 8;
        }
        data
    }

    fn manual_unpack(&mut self, data: &[u8]) {
        assert_eq!(data.len() % STRUCT_SIMPLE_PACKED_SIZE, 0);
        assert_eq!(data.len() / STRUCT_SIMPLE_PACKED_SIZE, self.0.len());
        let mut pos = 0;
        for elem in &mut self.0 {
            elem.a = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.b = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.c = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.d = f64::from_be_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
        }
    }
}

pub struct StructSimpleState {
    data: *mut Vec<StructSimple>,
}

impl PackedSize for StructSimpleState {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok((*self.data).len() * STRUCT_SIMPLE_PACKED_SIZE)
    }
}

impl PackMethod for StructSimpleState {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        let mut i = offset / STRUCT_SIMPLE_PACKED_SIZE;
        let mut pos: isize = 0;
        while i < (*self.data).len() && (pos + STRUCT_SIMPLE_PACKED_SIZE as isize) <= dst_size as isize {
            let elem = &(*self.data)[i];
            std::ptr::copy_nonoverlapping(elem.a.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.b.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.c.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.d.to_be_bytes().as_ptr(), dst.offset(pos), 8);
            pos += 8;
            i += 1;
        }
        Ok(pos.try_into().expect("failed to convert pos from isize to usize"))
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>> {
        Ok(vec![])
    }
}

impl UnpackMethod for StructSimpleState {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        // Assume we get a full non-fragmented buffer for now.
        assert_eq!(offset, 0);
        let used = (*self.data).len() * STRUCT_SIMPLE_PACKED_SIZE;
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
        Ok(vec![])
    }
}

/// Packed size for the simple struct without gap.
pub const STRUCT_SIMPLE_NO_GAP_PACKED_SIZE: usize = 2 * std::mem::size_of::<i32>() + std::mem::size_of::<f64>();

#[derive(Equivalence)]
#[repr(C)]
pub struct StructSimpleNoGap {
    a: i32,
    b: i32,
    c: f64,
}

impl StructSimpleNoGap {
    pub fn new() -> StructSimpleNoGap {
        StructSimpleNoGap {
            a: 34,
            b: -2332,
            c: 1.9,
        }
    }
}

pub struct StructSimpleNoGapArray(pub Vec<StructSimpleNoGap>);

impl StructSimpleNoGapArray {
    pub fn new(size: usize) -> StructSimpleNoGapArray {
        assert_eq!(size % STRUCT_SIMPLE_NO_GAP_PACKED_SIZE, 0);
        assert!(size >= STRUCT_SIMPLE_NO_GAP_PACKED_SIZE);
        let count = size / STRUCT_SIMPLE_NO_GAP_PACKED_SIZE;
        StructSimpleNoGapArray((0..count).map(|_| StructSimpleNoGap::new()).collect())
    }

    pub fn update(&mut self, size: usize) {
        assert_eq!(size % STRUCT_SIMPLE_NO_GAP_PACKED_SIZE, 0);
        assert!(size >= STRUCT_SIMPLE_NO_GAP_PACKED_SIZE);
        let count = size / STRUCT_SIMPLE_NO_GAP_PACKED_SIZE;
        assert!(self.0.len() < count);
        let new = count - self.0.len();
        for _ in 0..new {
            self.0.push(StructSimpleNoGap::new());
        }
    }
}

impl MessageCount for StructSimpleNoGapArray {
    fn count(&self) -> usize {
        self.0.len()
    }
}

impl MessagePointer for StructSimpleNoGapArray {
    fn ptr(&self) -> *const u8 {
        self.0.as_ptr() as *const _
    }

    fn ptr_mut(&mut self) -> *mut u8 {
        self.0.as_mut_ptr() as *mut _
    }
}

impl MessageBuffer for StructSimpleNoGapArray {
    unsafe fn pack(&self) -> Option<DatatypeResult<Box<dyn PackMethod>>> {
        Some(Ok(Box::new(StructSimpleNoGapState {
            data: (&self.0 as *const Vec<StructSimpleNoGap>) as *mut _,
        })))
    }

    unsafe fn unpack(&mut self) -> Option<DatatypeResult<Box<dyn UnpackMethod>>> {
        Some(Ok(Box::new(StructSimpleNoGapState {
            data: &mut self.0,
        })))
    }
}

impl ManualPack for StructSimpleNoGapArray {
    fn packed_size(&self) -> usize {
        self.0.len() * STRUCT_SIMPLE_NO_GAP_PACKED_SIZE
    }

    fn manual_pack(&self) -> Vec<u8> {
        let mut data = vec![0; self.0.len() * STRUCT_SIMPLE_NO_GAP_PACKED_SIZE];
        let mut pos = 0;
        for elem in &self.0 {
            let _ = &data[pos..pos+4].copy_from_slice(&elem.a.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+4].copy_from_slice(&elem.b.to_be_bytes());
            pos += 4;
            let _ = &data[pos..pos+8].copy_from_slice(&elem.c.to_be_bytes());
            pos += 8;
        }
        data
    }

    fn manual_unpack(&mut self, data: &[u8]) {
        assert_eq!(data.len() % STRUCT_SIMPLE_NO_GAP_PACKED_SIZE, 0);
        assert_eq!(data.len() / STRUCT_SIMPLE_NO_GAP_PACKED_SIZE, self.0.len());
        let mut pos = 0;
        for elem in &mut self.0 {
            elem.a = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.b = i32::from_be_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            elem.c = f64::from_be_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
        }
    }
}

pub struct StructSimpleNoGapState {
    data: *mut Vec<StructSimpleNoGap>,
}

impl PackedSize for StructSimpleNoGapState {
    unsafe fn packed_size(&self) -> DatatypeResult<usize> {
        Ok((*self.data).len() * STRUCT_SIMPLE_NO_GAP_PACKED_SIZE)
    }
}

impl PackMethod for StructSimpleNoGapState {
    unsafe fn pack(&mut self, offset: usize, dst: *mut u8, dst_size: usize) -> DatatypeResult<usize> {
        let mut i = offset / STRUCT_SIMPLE_NO_GAP_PACKED_SIZE;
        let mut pos: isize = 0;
        while i < (*self.data).len() && (pos + STRUCT_SIMPLE_NO_GAP_PACKED_SIZE as isize) <= dst_size as isize {
            let elem = &(*self.data)[i];
            std::ptr::copy_nonoverlapping(elem.a.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.b.to_be_bytes().as_ptr(), dst.offset(pos), 4);
            pos += 4;
            std::ptr::copy_nonoverlapping(elem.c.to_be_bytes().as_ptr(), dst.offset(pos), 8);
            pos += 8;
            i += 1;
        }
        Ok(pos.try_into().expect("failed to convert pos from isize to usize"))
    }

    unsafe fn memory_regions(&self) -> DatatypeResult<Vec<(*const u8, usize)>> {
        Ok(vec![])
    }
}

impl UnpackMethod for StructSimpleNoGapState {
    unsafe fn unpack(&mut self, offset: usize, src: *const u8, src_size: usize) -> DatatypeResult<()> {
        // Assume we get a full non-fragmented buffer for now.
        assert_eq!(offset, 0);
        let used = (*self.data).len() * STRUCT_SIMPLE_NO_GAP_PACKED_SIZE;
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
            std::ptr::copy_nonoverlapping(src.offset(pos), tmp.as_mut_ptr(), 8);
            pos += 8;
            elem.c = f64::from_be_bytes(tmp[..8].try_into().unwrap());
        }
        Ok(())
    }

    unsafe fn memory_regions(&mut self) -> DatatypeResult<Vec<(*mut u8, usize)>> {
        Ok(vec![])
    }
}
