use std::mem::MaybeUninit;
use std::ffi::{c_void, c_char};
use mpicd_pmix_sys::{PMIx_Initialized, PMIx_Init, PMIx_Get, PMIx_Value_unload, pmix_proc_t, pmix_value_t, PMIX_SUCCESS, PMIX_LOCAL_RANK, PMIX_NODE_RANK, PMIX_UINT16};
use log::info;

unsafe fn get_u16(proc: pmix_proc_t, key: *const c_char) -> u16 {
    let mut value = MaybeUninit::<*mut pmix_value_t>::uninit();
    let ret = PMIx_Get(&proc, key, std::ptr::null_mut(), 0, value.as_mut_ptr());
    if ret != (PMIX_SUCCESS as i32) {
        panic!("PMIx_Get failed");
    }
    let value = value.assume_init();
    if value == std::ptr::null_mut() {
        panic!("PMIx_Get returned NULL value");
    } else if (*value).type_ != (PMIX_UINT16 as u16) {
        panic!("PMIx_Get returned wrong type");
    }
    let i: *mut u16 = Box::into_raw(Box::new(0));
    let mut size = 0;
    PMIx_Value_unload(value, &mut (i as *mut c_void), &mut size);
    *Box::from_raw(i)
}

pub unsafe fn init() {
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

    let local_rank = get_u16(proc, PMIX_LOCAL_RANK.as_ptr() as *const _);
    info!("local_rank = {}", local_rank);

    let node_rank = get_u16(proc, PMIX_NODE_RANK.as_ptr() as *const _);
    info!("node_rank = {}", node_rank);
}
