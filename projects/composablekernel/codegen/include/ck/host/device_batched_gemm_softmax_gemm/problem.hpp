// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

// defines the problem specification for a GEMM operation
struct Problem
{
    std::size_t M             = 0;
    std::size_t N             = 0;
    std::size_t K             = 0;
    std::size_t O             = 0;
    bool TransA               = false;
    bool TransB               = false;
    bool TransB1              = false;
    bool TransC               = false;
    DataType ADataType        = DataType::Half;
    DataType BDataType        = DataType::Half;
    DataType B1DataType       = DataType::Half;
    DataType CDataType        = DataType::Half;
    std::string AElementOp    = PassThrough;
    std::string BElementOp    = PassThrough;
    std::string B1ElementOp   = PassThrough;
    std::string CElementOp    = PassThrough;
    std::string AccElementOp  = Scale;
    bool MaskOutUpperTriangle = false;

    // returns the correct device op file for the operation
    std::string GetIncludeHeader() const;

    // returns a list of instances based on the problem spec and provided fusion operations
    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue = "",
                                       const std::string& epilogue = "") const;
};

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
