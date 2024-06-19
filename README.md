# mpicd

Implements custom datatype API for MPI.

## Dependencies

Required dependencies:

* Working Rust install (take a look at <https://www.rust-lang.org/tools/install> for instructions)
* Clang/LLVM install (libclang is needed for Rust's bindgen code)
* UCX installation (>=1.12) (<https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz>)
* PMIx installation (>=5.x) (<https://github.com/openpmix/openpmix/releases/download/v5.0.2/pmix-5.0.2.tar.gz>)
* CMake (>=3.22)
* Working Open MPI install (mpirun will be used to launch the examples/benchmarks) --- the
  latest main branch should work

Make sure to set `PKG_CONFIG_PATH` properly for the UCX and PMIx installs,
since both the CMake build and Rust builds rely on this.

## Building

You should be able to build this with the standard cmake build process,
assuming all dependencies are installed correctly:

```
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=ReleaseWithDebInfoOptFull -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX ..
make install
```

This should install the library and mpicc/mpicxx scripts in `$INSTALL_PREFIX/bin`.

### Linking errors

If you run into linking errors related to UCX or PMIx, then you may need to set
the rpath manually with the RUSTFLAGS environment variable when running make:
`RUSTFLAGS="-C link-args=-Wl,-rpath,$UCX_OR_PMIX_INSTALL_PATH/lib" make install`.

## Running the examples

The examples can be run with Open MPI's `mpirun` with a maximum of two ranks
(these don't have to be on the same node).

### Rust benchmarks

The Rust benchmarks must be built directly with cargo, by running
`cargo build --release`. The binaries will be placed in `target/release/`.
