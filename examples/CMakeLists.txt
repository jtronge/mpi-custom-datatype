add_executable(hello_world hello_world.c)
add_executable(datatype0 datatype0.c)

foreach(BIN hello_world datatype0)
    target_link_libraries(${BIN} PUBLIC mpicd-capi)
    target_link_libraries(${BIN} PUBLIC ${UCX_LINK_LIBRARIES})
    target_link_libraries(${BIN} PUBLIC ${PMIX_LINK_LIBRARIES})
endforeach()
