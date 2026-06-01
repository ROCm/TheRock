// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "matrix_core_swizzle_kernel.hpp"
#include <string>

struct matrix_core_swizzle_traits
{
    std::string data_type; // fp16 only
    std::string inst;      // 32x32x8, 16x16x16
    std::string permute;   //
};

using matrix_core_swizzle_args = matrix_core_swizzle_host_args;

// host API
float matrix_core_swizzle(matrix_core_swizzle_traits,
                          matrix_core_swizzle_args,
                          const ck_tile::stream_config&);
