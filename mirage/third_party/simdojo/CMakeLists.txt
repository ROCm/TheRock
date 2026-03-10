# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.

add_library(simdojo OBJECT
    component.cpp
    simulation.cpp
    topology.cpp
)
set_target_properties(simdojo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(simdojo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(simdojo PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(simdojo PUBLIC Threads::Threads)
