// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

// defines the problem specification for a GEMM operation
struct Problem
{
    // dimensions for GEMM operation
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t K = 0;
    // layouts for tensors
    bool TransA                      = false;
    bool TransB                      = false;
    bool TransE                      = false;
    std::vector<bool> DsTrans        = {};
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType EDataType               = DataType::Half;
    std::vector<DataType> DsDataType = {};
    std::string AElementOp           = PassThrough;
    std::string BElementOp           = PassThrough;
    std::string CDEElementOp         = PassThrough;

    // returns the correct device op file for the operation
    std::string GetIncludeHeader() const;

    // returns a list of instances based on the problem spec and provided fusion operations
    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue = "",
                                       const std::string& epilogue = "") const;
};

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
