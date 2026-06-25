# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(utility
    device_memory.cpp
    host_tensor.cpp
    convolution_parameter.cpp
)

add_library(composable_kernel::utility ALIAS utility)
set_target_properties(utility PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(utility
    PROPERTIES
    VERSION ${CMAKE_PROJECT_VERSION}
    SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR})
target_compile_options(utility PRIVATE ${CMAKE_COMPILER_WARNINGS})
target_include_directories(utility PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>"
)
if(WIN32)
    target_compile_definitions(utility PUBLIC NOMINMAX)
endif()

rocm_install(
    TARGETS utility
    EXPORT utilityTargets
)

rocm_install(
    EXPORT utilityTargets
    FILE composable_kernelutilityTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

clang_tidy_check(utility)
