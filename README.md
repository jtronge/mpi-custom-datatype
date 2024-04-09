# mpicd

Implements custom datatype API for MPI.

## Dependencies

Required dependencies:

* Working Rust install (take a look at <https://www.rust-lang.org/tools/install> for instructions)
* Clang/LLVM install (libclang is needed for Rust's bindgen code)
* UCX installation (>=1.12) (<https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz>)
* PMIx installation (>=5.x) (<https://github.com/openpmix/openpmix/releases/download/v5.0.2/pmix-5.0.2.tar.gz>)
* CMake (>=3.22)

Make sure to set `PKG_CONFIG_PATH` properly for the UCX and PMIx installs,
since both the CMake build and Rust builds rely on this.

## Building

You should be able to build this with the standard cmake build process,
assuming all dependencies are installed correctly.

## Examples

...

## Rust log output

To see debug output from the Rust code, set `RUST_LOG=debug` in your
environment.
