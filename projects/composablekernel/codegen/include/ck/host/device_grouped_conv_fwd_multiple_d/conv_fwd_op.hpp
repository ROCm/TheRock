// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/operation/gemm.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp"

namespace ck {
namespace host {
namespace conv {

// defines the values needed for an instance of forward convolution and functions to return
// (templated) instances
struct Operation_Conv_Fwd_Xdl_Cshuffle
{
    // returns a vector of instances given the fusion operations, uses default values for problem
    // spec
    static std::vector<Operation_Conv_Fwd_Xdl_Cshuffle>
    CreateOperations(const std::string& prologue, const std::string& epilogue);
    // returns a vector of instances, provided with a problem spec and fusion operations
    static std::vector<Operation_Conv_Fwd_Xdl_Cshuffle> CreateOperations(
        const Problem_Conv_Fwd& prob, const std::string& prologue, const std::string& epilogue);
    std::size_t NumDim;
    TensorDesc A{};
    TensorDesc B{};
    DataType acc               = DataType::Float;
    DataType cs_type           = DataType::Half;
    std::vector<TensorDesc> Ds = {};
    TensorDesc E{};
    std::string a_elem_op   = PassThrough;
    std::string b_elem_op   = PassThrough;
    std::string cde_elem_op = PassThrough;
    std::string prologue    = "";
    std::string epilogue    = "";
    std::string conv_specialization =
        "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default";
    std::string gemm_specialization =
        "ck::tensor_operation::device::GemmSpecialization::MNKPadding";
    // tuning parameters
    operation::TileDesc tile_desc{};
    operation::BlockTransferDesc a_block_transfer{};
    operation::BlockTransferDesc b_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc c_block_transfer{};

    // functions to update fusion operations if they are provided
    void update_prologue(const std::string& prologue);
    void update_epilogue(const std::string& epilogue);
    // returns a templated instance
    Solution ToSolution() const;
};

} // namespace conv
} // namespace host
} // namespace ck
