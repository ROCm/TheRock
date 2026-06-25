# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test_executable(test_scatter_gather scatter_gather.cpp)
# target_compile_options(test_scatter_gather PRIVATE -v --save-temps -Wno-gnu-line-marker)
