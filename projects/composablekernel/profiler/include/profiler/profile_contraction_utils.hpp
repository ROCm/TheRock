// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck/ck.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

enum struct ContractionMatrixLayout
{
    MK_KN_MN_MN, // 0
    MK_NK_MN_MN, // 1
    KM_KN_MN_MN, // 2
    KM_NK_MN_MN, // 3
};

enum struct ContractionDataType
{
    F32_F32_F32_F32,     // 0
    F64_F64_F64_F64,     // 1
    F16_F16_F16_F16,     // 2
    BF16_BF16_BF16_BF16, // 3
};

enum struct ContractionComputeDataType
{
    F32 = 0,
    F64,
    F16,
    BF16,
};

inline void collect_index_params(char* argv[],
                                 std::vector<ck::index_t>& params,
                                 const ck::index_t from,
                                 const ck::index_t num)
{
    for(ck::index_t p = from; p < from + num; p++)
        params.push_back(std::stoi(argv[p]));
}

// Defualt strides for row-major: {Dim1 * Dim2 * Dim3, Dim2 * Dim3, Dim3, 1}
// Defualt strides for column-major: {Dim1, 1, Dim0 * Dim1 * Dim3, Dim0 * Dim1}

// M1, 1, M0 * M1 * K1, M0 * M1
// K0, K1, M0, M1
inline void
assign_default_strides(Row, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    ck::index_t stride = 1;
    for(ck::index_t s = strides.size() - 1; s >= 0; s--)
    {
        strides[s] = stride;
        stride *= dims[s];
    }
}

inline void
assign_default_strides(Col, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    // Assign second half of strides
    ck::index_t stride = 1;
    for(ck::index_t s = strides.size() / 2 - 1; s >= 0; s--)
    {
        strides[s] = stride;
        stride *= dims[s];
    }

    // Assign first half of strides
    for(ck::index_t s = strides.size() - 1; s > static_cast<ck::index_t>(strides.size()) / 2 - 1;
        s--)
    {
        strides[s] = stride;
        stride *= dims[s];
    }
}
