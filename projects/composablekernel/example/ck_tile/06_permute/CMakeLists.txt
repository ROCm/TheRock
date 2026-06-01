# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
add_executable(tile_example_permute permute.cpp)

if(NOT DEFINED PERMUTE_USE_ALTERNATIVE_IMPL)
# set(PERMUTE_USE_ALTERNATIVE_IMPL false)
set(PERMUTE_USE_ALTERNATIVE_IMPL true)
endif()
if(PERMUTE_USE_ALTERNATIVE_IMPL)
target_compile_options(tile_example_permute PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
target_sources(tile_example_permute PRIVATE alternative_impl/matrix_core_swizzle.cpp)
endif()
# target_compile_options(tile_example_permute PRIVATE -v --save-temps -Wno-gnu-line-marker)
