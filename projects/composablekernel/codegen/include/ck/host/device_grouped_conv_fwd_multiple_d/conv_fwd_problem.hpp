// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace conv {

// defines the problem specification for a forward convolution operation
struct Problem_Conv_Fwd
{
    std::size_t NumDim = 0;
    // size of a forward convolution operation
    std::size_t G                    = 0;
    std::size_t N                    = 0;
    std::size_t C                    = 0;
    std::size_t Hi                   = 0;
    std::size_t Wi                   = 0;
    std::size_t Ho                   = 0;
    std::size_t Wo                   = 0;
    std::size_t K                    = 0;
    std::size_t Y                    = 0;
    std::size_t X                    = 0;
    Layout ALayout                   = Layout::NHWGC;
    Layout BLayout                   = Layout::GKYXC;
    Layout ELayout                   = Layout::NHWGK;
    std::vector<Layout> DsLayout     = {};
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType EDataType               = DataType::Half;
    std::vector<DataType> DsDataType = {};
    std::string AElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string CDEElementOp         = "ck::tensor_operation::element_wise::PassThrough";

    // returns the correct device op file for the operation
    std::string GetIncludeHeader() const;

    // returns a list of instances based on the problem spec and provided fusion operations
    std::vector<Solution> GetSolutions(const std::string& arch,
                                       const std::string& prologue,
                                       const std::string& epilogue) const;
};

} // namespace conv
} // namespace host
} // namespace ck
