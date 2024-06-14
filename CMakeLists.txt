cmake_minimum_required(VERSION 3.22)
project(mpicd)

# Set up Corrosion to interface with the Rust code
include(FetchContent)
FetchContent_Declare(Corrosion GIT_REPOSITORY git@github.com:corrosion-rs/corrosion.git GIT_TAG v0.4.8)
FetchContent_MakeAvailable(Corrosion)

find_package(PkgConfig REQUIRED)
pkg_check_modules(UCX REQUIRED ucx>=1.12)
pkg_check_modules(PMIX REQUIRED pmix>=5)

# Add the Rust library
include_directories(mpicd-capi/include)
corrosion_import_crate(MANIFEST_PATH mpicd-capi/Cargo.toml PROFILE release)
target_sources(mpicd-capi PUBLIC mpicd-capi/include/mpi.h)

# Configure the mpicc script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpicc.in
               ${CMAKE_CURRENT_BINARY_DIR}/scripts/mpicc)

add_subdirectory(examples)

# Install the rust library (this is a hack, for some reason corrosion doesn't
# support library installs)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libmpicd_capi.a
        DESTINATION lib)
# Install mpicc
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/scripts/mpicc
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        DESTINATION bin)
