//! Code abstracting out Rust communicators.
use crate::Status;
use crate::datatype::MessageBuffer;

#[derive(Copy, Clone, Debug)]
pub enum Error {
    /// An internal error occured.
    InternalError,

    /// No message was found during a probe operation.
    NoProbeMessage,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Result of a probe call.
#[derive(Copy, Clone, Debug)]
pub struct ProbeResult {
    /// Number of bytes.
    pub size: usize,

    /// Source process.
    pub source: i32,
}

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
    unsafe fn isend<B: MessageBuffer + ?Sized>(&self, data: &B, dest: i32, tag: i32) -> Result<Self::Request>;

    /// Do a non-blocking recv of data from the source with the specified tag.
    unsafe fn irecv<B: MessageBuffer + ?Sized>(&self, data: &mut B, source: i32, tag: i32) -> Result<Self::Request>;

    /// Probe for an incoming message.
    fn probe(&self, source: Option<i32>, tag: i32) -> Result<ProbeResult>;

    /// Wait for all requests in list to complete.
    unsafe fn waitall(&self, requests: &[Self::Request]) -> Result<Vec<Status>>;
}
