//! C context data management code.

/// C context struct to hold additional context data specific to the C interface.
pub(crate) struct CContext {}

impl CContext {
    pub(crate) fn new() -> CContext {
        CContext {}
    }
}
