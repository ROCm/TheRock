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
namespace device_gemm_multiple_d {

struct Problem
{
    std::size_t M                    = 0;
    std::size_t N                    = 0;
    std::size_t K                    = 0;
    bool TransA                      = false;
    bool TransB                      = false;
    bool TransE                      = false;
    std::vector<bool> DsTrans        = {};
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType EDataType               = DataType::Half;
    std::vector<DataType> DsDataType = {};
    std::string AElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string CDEElementOp         = "ck::Tuple<>";

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
