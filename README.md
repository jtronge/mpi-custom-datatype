# mpicd

Implements custom datatype API for MPI.

## Dependencies

Required dependencies:

* Working recent Rust install (take a look at <https://www.rust-lang.org/tools/install> for instructions)
* UCX installation (>=1.12) (<https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz>)
* PMIx installation (>=5.x) (<https://github.com/openpmix/openpmix/releases/download/v5.0.2/pmix-5.0.2.tar.gz>)
* CMake (>=3.22)

Make sure to set `PKG_CONFIG_PATH` properly for the UCX and PMIx installs,
since both the CMake build and Rust builds rely on this.

## Building

You should be able to build this with the standard cmake build process,
assuming all dependencies are installed correctly.

## Examples
