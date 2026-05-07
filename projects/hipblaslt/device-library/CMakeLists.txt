# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(TENSILELITE_BUILD_PARALLEL_LEVEL "" CACHE STRING "Number of CPU cores to use for building device libraries (will use nproc if unset).")
set(TENSILELITE_KEEP_BUILD_TMP OFF CACHE STRING "Keep temporary build directory for device libraries (turning this ON  bloat the build size).")
set(TENSILELITE_LIBRARY_FORMAT "" CACHE STRING "Format of master solution library files (msgpack or yaml).")
set(TENSILELITE_ASM_DEBUG "" CACHE STRING "Keep debug information for built code objects.")
set(TENSILELITE_LOGIC_FILTER "" CACHE STRING "Cutomsized logic filter, default is *, i.e. all logics.")
set(TENSILELITE_NO_COMPRESS "" CACHE STRING "Do not compress device code object files.")
set(TENSILELITE_EXPERIMENTAL "" CACHE STRING "Process experimental logic files.")
set(TENSILELITE_ENABLE_ASM_COMMENTS OFF CACHE STRING "Turn on asm comments when building gemm libraries.")

set(HIPBLASLT_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
set(HIPBLASLT_TENSILE_LIBPATH "${hipblaslt_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")
file(MAKE_DIRECTORY "${HIPBLASLT_TENSILE_LIBPATH}/library")

list(JOIN GPU_TARGETS "$<SEMICOLON>" TENSILELITE_GPU_TARGETS_SEMI_ESCAPED)

set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--architecture=${TENSILELITE_GPU_TARGETS_SEMI_ESCAPED}")
set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--cxx-compiler=${CMAKE_CXX_COMPILER}")
if(HIPBLASLT_ENABLE_ASAN)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--address-sanitizer")
endif()
if(TENSILELITE_BUILD_PARALLEL_LEVEL)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--jobs=${TENSILELITE_BUILD_PARALLEL_LEVEL}")
endif()
if(TENSILELITE_KEEP_BUILD_TMP)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--keep-build-tmp")
endif()
if(TENSILELITE_ASM_DEBUG)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--asm-debug")
endif()
if(TENSILELITE_LIBRARY_FORMAT)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--library-format=${TENSILELITE_LIBRARY_FORMAT}")
endif()
if(TENSILELITE_LOGIC_FILTER)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--logic-filter=${TENSILELITE_LOGIC_FILTER}")
endif()
if(TENSILELITE_NO_COMPRESS)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--no-compress")
endif()
if(TENSILELITE_EXPERIMENTAL)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--experimental")
endif()
if(NOT HIPBLASLT_ENABLE_LAZY_LOAD)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--no-lazy-library-loading")
endif()
if(NOT TENSILELITE_ENABLE_ASM_COMMENTS)
    set(TENSILELITE_BUILD_OPTS ${TENSILELITE_BUILD_OPTS} "--disable-asm-comments")
endif()

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()

set(output_stamp "${CMAKE_CURRENT_BINARY_DIR}/Tensile.stamp")
set(TENSILE_CREATE_LIBRARY_COMMAND
    ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
    ${TENSILELITE_BUILD_OPTS}
    ${HIPBLASLT_LIBLOGIC_PATH}
    "${HIPBLASLT_TENSILE_LIBPATH}"
    HIP
)

add_custom_command(
    OUTPUT "${output_stamp}"
    COMMENT "Building device libraries to ${HIPBLASLT_TENSILE_LIBPATH} ..."
    COMMAND ${TENSILE_CREATE_LIBRARY_COMMAND}
    COMMAND ${CMAKE_COMMAND} -E touch "${output_stamp}"
    DEPENDS  ${HIPBLASLT_PYTHON_DEPS}
    # Because the command can contain special characters
    VERBATIM
    # Because this can be very long running and difficult to debug deadlocks
    # without streaming.
    USES_TERMINAL
)

block(SCOPE_FOR VARIABLES)
    list(JOIN TENSILE_CREATE_LIBRARY_COMMAND " " FORMATTED_TCL)
    message(STATUS "Device lib build command: ${FORMATTED_TCL}")
endblock()

add_custom_target(tensilelite-device-libraries ALL
    DEPENDS "${output_stamp}"
)

add_subdirectory(extops)
add_subdirectory(matrix-transform)
