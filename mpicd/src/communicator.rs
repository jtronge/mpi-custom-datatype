//! Code abstracting out Rust communicators.
use crate::Status;
use crate::datatype::MessageBuffer;

#[derive(Copy, Clone, Debug)]
pub enum Error {
    InternalError,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Trait implementing simple p2p communication on top of some lower-level library.
pub trait Communicator {
    type Request;

    /// Return the number of processes in this communicator.
    fn size(&self) -> i32;

    /// Return the current rank of the process.
    fn rank(&self) -> i32;

    /// Perform a barrier on the processes.
    fn barrier(&self);

    /// Do a non-blocking send of data to the destination with specified tag.
    unsafe fn isend<B: MessageBuffer>(&self, data: B, dest: i32, tag: i32) -> Result<Self::Request>;

    /// Do a non-blocking recv of data from the source with the specified tag.
    unsafe fn irecv<B: MessageBuffer>(&self, data: B, source: i32, tag: i32) -> Result<Self::Request>;

    /// Wait for all requests in list to complete.
    unsafe fn waitall(&self, requests: &[Self::Request]) -> Result<Vec<Status>>;
}
