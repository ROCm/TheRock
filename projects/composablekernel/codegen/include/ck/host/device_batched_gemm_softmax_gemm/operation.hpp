// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/operation/gemm.hpp"
#include "ck/host/device_batched_gemm_softmax_gemm/problem.hpp"

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

// defines all values need for an instance of fwd conv
struct Operation_Xdl_CShuffle
{
    // returns a vector of instances, only given fusion operators: will use default problem spec
    static std::vector<std::vector<Operation_Xdl_CShuffle>>
    CreateOperations(const std::string& prologue, const std::string& epilogue);
    // returns a vector of instances, given a problem spec and fusion operators
    static std::vector<Operation_Xdl_CShuffle>
    CreateOperations(const Problem& prob, const std::string& prologue, const std::string& epilogue);
    TensorDesc A{};
    TensorDesc B{};
    TensorDesc B1{};
    TensorDesc C{};
    DataType acc                    = DataType::Float;
    DataType cs_type                = DataType::Half;
    std::string a_elem_op           = PassThrough;
    std::string b_elem_op           = PassThrough;
    std::string b1_elem_op          = PassThrough;
    std::string c_elem_op           = PassThrough;
    std::string acc_elem_op         = Scale;
    std::string prologue            = "";
    std::string epilogue            = "";
    std::string gemm_specialization = "ck::tensor_operation::device::GemmSpecialization::Default";
    // tuning parameters
    operation::TileDescGemmGemm tile_desc{};
    operation::BlockTransferDesc a_block_transfer{};
    operation::BlockTransferDesc b0_block_transfer{};
    operation::BlockTransferDesc b1_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc c_block_transfer{};

    bool mask_out_upper_triangle = false;

    // functions to update fusion operators if provided
    void update_prologue(const std::string& prologue);
    void update_epilogue(const std::string& epilogue);
    /**constexpr**/ bool
    IsSupported(std::size_t MRaw_, std::size_t NRaw_, std::size_t KRaw_, std::size_t Gemm1NRaw_);
    // returns a templated instance
    Solution ToSolution() const;
};

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
