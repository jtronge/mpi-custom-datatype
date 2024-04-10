//! C context data management code.
use std::ffi::{c_int, c_void};
use mpicd::datatype::{SendBuffer, RecvBuffer};
use crate::{datatype::CustomDatatype, consts, c};

/// C context struct to hold additional context data specific to the C interface.
pub(crate) struct CContext {
    datatypes: Vec<CustomDatatype>,
}

impl CContext {
    pub(crate) fn new() -> CContext {
        CContext {
            datatypes: vec![],
        }
    }

    /// Add a new datatype, returning it's C datatype integer.
    pub(crate) fn add_custom_datatype(&mut self, datatype: CustomDatatype) -> c::Datatype {
        let id = TryInto::<c_int>::try_into(self.datatypes.len()).unwrap() + consts::MAX_PREDEFINED + 1;
        self.datatypes.push(datatype);
        id
    }

    pub(crate) fn get_custom_datatype(&self, datatype: c::Datatype) -> Option<CustomDatatype> {
        if datatype <= consts::BYTE {
            None
        } else {
            let i: usize = (datatype - consts::MAX_PREDEFINED - 1).try_into().unwrap();
            self.datatypes.get(i).copied()
        }
    }
}
