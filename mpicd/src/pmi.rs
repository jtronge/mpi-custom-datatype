use std::mem::MaybeUninit;
use std::ffi::{c_void, c_char, CString};
use mpicd_pmix_sys::{
    PMIx_Initialized, PMIx_Init, PMIx_Finalize, PMIx_Put, PMIx_Get, PMIx_Commit, PMIx_Value_unload,
    PMIx_Fence, pmix_proc_t, pmix_value_t, pmix_value__bindgen_ty_1,
    pmix_byte_object_t, PMIX_SUCCESS, PMIX_GLOBAL, PMIX_LOCAL_RANK, PMIX_NODE_RANK,
    PMIX_JOB_SIZE, PMIX_UNIV_SIZE, PMIX_JOB_NUM_APPS,
    PMIX_LOCAL_SIZE, PMIX_UINT16, PMIX_UINT32, PMIX_BYTE_OBJECT,
};
use log::info;

trait PMIXType {
    unsafe fn unload(value: *mut pmix_value_t) -> Self;
    unsafe fn load(&mut self) -> pmix_value_t;
}

macro_rules! make_pmix_type {
    ($type:path, $constant:path) => {
        impl PMIXType for $type {
            unsafe fn unload(value: *mut pmix_value_t) -> Self {
                if (*value).type_ != ($constant as u16) {
                    panic!("PMIx_Get returned wrong type");
                }
                let i: *mut $type = Box::into_raw(Box::new(0));
                let mut size = 0;
                PMIx_Value_unload(value, &mut (i as *mut c_void), &mut size);
                *Box::from_raw(i)
            }

            unsafe fn load(&mut self) -> pmix_value_t {
                panic!("Not implemented");
            }
        }
    };
}

make_pmix_type!(u16, PMIX_UINT16);
make_pmix_type!(u32, PMIX_UINT32);

impl PMIXType for Vec<u8> {
    unsafe fn unload(value: *mut pmix_value_t) -> Self {
        let ptr = (*value).data.bo.bytes as *const u8;
        let size = (*value).data.bo.size;
        let mut out = Vec::with_capacity(size);
        std::ptr::copy(ptr, out.as_mut_ptr(), size);
        out.set_len(size);
        out
    }

    unsafe fn load(&mut self) -> pmix_value_t {
        pmix_value_t {
            type_: PMIX_BYTE_OBJECT as u16,
            data: pmix_value__bindgen_ty_1 {
                bo: pmix_byte_object_t {
                    bytes: self.as_mut_ptr() as *mut _,
                    size: self.len(),
                }
            }
        }
    }
}

unsafe fn get<T: PMIXType>(proc: pmix_proc_t, key: *const c_char) -> T {
    let mut value = MaybeUninit::<*mut pmix_value_t>::uninit();
    let ret = PMIx_Get(&proc, key, std::ptr::null_mut(), 0, value.as_mut_ptr());
    if ret != (PMIX_SUCCESS as i32) {
        panic!("PMIx_Get failed");
    }
    let value = value.assume_init();
    if value == std::ptr::null_mut() {
        panic!("PMIx_Get returned NULL value");
    }
    T::unload(value)
}

pub struct PMI {
    proc: pmix_proc_t,
    rank: u32,
    size: u32,
}

impl PMI {
    pub fn init() -> PMI {
        unsafe {
            if PMIx_Initialized() != 0 {
                panic!("PMIx was already initialized");
            }

            info!("Initializing PMIx");
            let mut proc = MaybeUninit::<pmix_proc_t>::uninit();
            let ret = PMIx_Init(proc.as_mut_ptr(), std::ptr::null_mut(), 0);
            if ret != (PMIX_SUCCESS as i32) {
                panic!("PMIx_Init failed");
            }
            let proc = proc.assume_init();
            info!("PMIx init successful");

            info!("proc.rank = {}", proc.rank);

            // TODO: Get PMIX_GLOBAL_RANK instead?

            let local_rank = get::<u16>(proc, PMIX_LOCAL_RANK.as_ptr() as *const _);
            info!("local_rank = {}", local_rank);

            let node_rank = get::<u16>(proc, PMIX_NODE_RANK.as_ptr() as *const _);
            info!("node_rank = {}", node_rank);

            let univ_size = get::<u32>(proc, PMIX_UNIV_SIZE.as_ptr() as *const _);
            info!("univ_size = {}", univ_size);

            let local_size = get::<u32>(proc, PMIX_LOCAL_SIZE.as_ptr() as *const _);
            info!("local_size = {}", local_size);

            PMI {
                proc,
                rank: (local_rank * node_rank) as u32,
                size: local_size,
            }
        }
    }

    #[inline]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Attempt to put a PMIXType with a given key (global by default).
    pub fn put<T: PMIXType>(&self, key: &str, mut value: T) {
        unsafe {
            let mut pmix_value = value.load();
            let s = CString::new(key).expect("failed to get CString for key");
            let ret = PMIx_Put(PMIX_GLOBAL as u8, s.as_ptr(), &mut pmix_value);
            if ret != (PMIX_SUCCESS as i32) {
                panic!("PMIx_Put failed");
            }

            // Just commit right away.
            let ret = PMIx_Commit();
            if ret != (PMIX_SUCCESS as i32) {
                panic!("PMIx_Commit failed");
            }
        }
    }

    /// Attempt to get a PMIXType from a process with the given key.
    pub fn get<T: PMIXType>(&self, rank: u32, key: &str) -> T {
        unsafe {
            let proc = pmix_proc_t {
                nspace: self.proc.nspace,
                rank,
            };
            let s = CString::new(key).expect("failed to get CString for key");
            get(proc, s.as_ptr())
        }
    }

    /// Perform a fence operation.
    pub fn fence(&self) {
        unsafe {
            let ret = PMIx_Fence(std::ptr::null(), 0, std::ptr::null(), 0);
            if ret != (PMIX_SUCCESS as i32) {
                panic!("PMIx_Fence failed");
            }
        }
    }
}

impl Drop for PMI {
    fn drop(&mut self) {
        unsafe {
            let ret = PMIx_Finalize(std::ptr::null(), 0);
            if ret != (PMIX_SUCCESS as i32) {
                panic!("PMIx_Finalize failed");
            }
        }
    }
}
