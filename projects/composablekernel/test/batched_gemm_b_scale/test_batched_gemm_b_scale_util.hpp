// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/data_type.hpp"
#include "profiler/profile_batched_gemm_b_scale_impl.hpp"

namespace ck {
namespace test {

template <typename Tuple>
class TestBatchedGemmBScale : public testing::Test
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using F32 = float;

    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using CLayout         = Row;
    using ADataType       = std::tuple_element_t<2, Tuple>;
    using BDataType       = std::tuple_element_t<3, Tuple>;
    using BScaleDataType  = std::tuple_element_t<4, Tuple>;
    using ComputeDataType = std::tuple_element_t<5, Tuple>;
    using CDataType       = std::tuple_element_t<6, Tuple>;

    public:
    static constexpr ck::index_t ScaleBlockK = 128; // all instances
    static constexpr bool verify_            = true;
    static constexpr int init_method_        = 2;
    static constexpr bool log_               = false;
    static constexpr bool bench_             = false; // measure kernel performance
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1, 2}; }

    void Run(const int M,
             const int N,
             const int K,
             const int StrideA,
             const int StrideB,
             const int StrideC,
             const int NBatch)
    {
        for(auto kb : k_batches_)
        {
            RunSingle(M, N, K, StrideA, StrideB, StrideC, NBatch, kb);
        }
    }

    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   const int Nbatch,
                   int kbatch   = 1,
                   int n_warmup = 1,
                   int n_iter   = 10)
    {
        const int BatchStrideA      = StrideA * M;
        const int BatchStrideB      = StrideB * K;
        const int BatchStrideC      = StrideC * M;
        const int BatchStrideScaleB = StrideB * K;
        bool pass                   = ck::profiler::profile_batched_gemm_b_scale_impl<ADataType,
                                                                                      BDataType,
                                                                                      BScaleDataType,
                                                                                      ComputeDataType,
                                                                                      F32,
                                                                                      CDataType,
                                                                                      ScaleBlockK,
                                                                                      ALayout,
                                                                                      BLayout,
                                                                                      CLayout>(verify_,
                                                                             init_method_,
                                                                             log_,
                                                                             bench_,
                                                                             M,
                                                                             N,
                                                                             K,
                                                                             StrideA,
                                                                             StrideB,
                                                                             StrideC,
                                                                             BatchStrideA,
                                                                             BatchStrideB,
                                                                             BatchStrideC,
                                                                             BatchStrideScaleB,
                                                                             Nbatch,
                                                                             kbatch,
                                                                             n_warmup,
                                                                             n_iter);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
