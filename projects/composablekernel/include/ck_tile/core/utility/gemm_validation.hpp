// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <stdexcept>
#include "ck_tile/core/config.hpp"

namespace ck_tile {

inline void
validate_stride(std::string Layout, int M, int N, int stride, const std::string& stride_name)
{
    if(Layout == "C" && stride < M)
    {
        throw std::runtime_error("For ColumnMajor layout, " + stride_name + "(" +
                                 std::to_string(stride) + ") must be greater or equal to dim " +
                                 std::to_string(M));
    }
    if(Layout == "R" && stride < N)
    {
        throw std::runtime_error("For RowMajor layout, " + stride_name + "(" +
                                 std::to_string(stride) + ") must be greater or equal to dim " +
                                 std::to_string(N));
    }
}

inline void validate_gemm_stride(std::string a_layout,
                                 std::string b_layout,
                                 std::string c_layout,
                                 int M,
                                 int N,
                                 int K,
                                 int Stride_A,
                                 int Stride_B,
                                 int Stride_C)
{
    // set default stride
    if(Stride_A <= 0)
        Stride_A = (a_layout == "R") ? K : M;
    if(Stride_B <= 0)
        Stride_B = (b_layout == "R") ? N : K;
    if(Stride_C <= 0)
        Stride_C = (c_layout == "R") ? N : M;

    validate_stride(a_layout, M, K, Stride_A, "Stride_A");
    validate_stride(b_layout, K, N, Stride_B, "Stride_B");
    validate_stride(c_layout, M, N, Stride_C, "Stride_C");
}
} // namespace ck_tile
