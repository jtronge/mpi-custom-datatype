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

add_subdirectory(examples)

# Hack to get full list of required libraries
get_target_property(MPICC_LINK_LIBRARIES hello_world LINK_LIBRARIES)

# Configure the mpicc script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpicc.in
               ${CMAKE_CURRENT_BINARY_DIR}/scripts/mpicc)

# Install the rust library (this is a hack, for some reason corrosion doesn't
# support library installs)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libmpicd_capi.so
        DESTINATION lib)
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
              ${CMAKE_INSTALL_PREFIX}/lib/libmpicd_capi.so \
              ${CMAKE_INSTALL_PREFIX}/lib/libmpi.so)")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libmpicd_capi.a
        RENAME libmpi.a
        DESTINATION lib)
# Should this be installed with a header-specific rule?
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mpicd-capi/include/mpi.h
        DESTINATION include)
# Install mpicc and variants
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/scripts/mpicc
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        DESTINATION bin)
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
              ${CMAKE_INSTALL_PREFIX}/bin/mpicc \
              ${CMAKE_INSTALL_PREFIX}/bin/mpicxx)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
              ${CMAKE_INSTALL_PREFIX}/bin/mpicc \
              ${CMAKE_INSTALL_PREFIX}/bin/mpiCC)")
