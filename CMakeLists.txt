cmake_minimum_required(VERSION 3.20)
project(mpicd)

# Set up Corrosion to interface with the Rust code
include(FetchContent)
FetchContent_Declare(Corrosion GIT_REPOSITORY git@github.com:corrosion-rs/corrosion.git GIT_TAG v0.4.8)
FetchContent_MakeAvailable(Corrosion)

find_package(PkgConfig REQUIRED)
pkg_check_modules(UCX REQUIRED ucx>=1.12)
pkg_check_modules(PMIX REQUIRED pmix>=5)

include_directories(mpicd-capi/include)
corrosion_import_crate(MANIFEST_PATH mpicd-capi/Cargo.toml PROFILE release)
add_executable(main main.c)
target_link_libraries(main PUBLIC mpicd-capi)
target_link_libraries(main PUBLIC ${UCX_LINK_LIBRARIES})
target_link_libraries(main PUBLIC ${PMIX_LINK_LIBRARIES})
