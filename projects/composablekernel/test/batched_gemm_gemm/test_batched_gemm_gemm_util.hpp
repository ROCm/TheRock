// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <vector>
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "profiler/profile_batched_gemm_gemm_impl.hpp"

using ck::tensor_operation::device::GemmSpecialization;

template <ck::index_t N>
using I = ck::Number<N>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
struct TestBatchedGemmGemm : public ::testing::Test
{
    using ADataType  = std::tuple_element_t<0, Tuple>;
    using B0DataType = std::tuple_element_t<1, Tuple>;
    using B1DataType = std::tuple_element_t<2, Tuple>;
    using CDataType  = std::tuple_element_t<3, Tuple>;
    using ALayout    = std::tuple_element_t<4, Tuple>;
    using B0Layout   = std::tuple_element_t<5, Tuple>;
    using B1Layout   = std::tuple_element_t<6, Tuple>;
    using CLayout    = std::tuple_element_t<7, Tuple>;

    std::vector<std::vector<int>> lengths_ = {
        {256, 256, 64, 64, 4},
        {256, 256, 128, 128, 4},
        {512, 512, 64, 64, 2},
        {512, 512, 128, 128, 2},
        {1024, 1024, 64, 64, 1},
        {1024, 1024, 128, 128, 1},
    };
    bool bench_  = false;
    bool verify_ = true;

    void RunSingle(int M, int N, int K, int O, int BatchCount)
    {
        bool pass = ck::profiler::profile_batched_gemm_gemm_impl<ADataType,
                                                                 B0DataType,
                                                                 B1DataType,
                                                                 CDataType,
                                                                 ALayout,
                                                                 B0Layout,
                                                                 B1Layout,
                                                                 CLayout>(
            verify_, 1, false, bench_, M, N, K, O, BatchCount);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M          = lengths[0];
            int N          = lengths[1];
            int K          = lengths[2];
            int O          = lengths[3];
            int BatchCount = lengths[4];

            this->RunSingle(M, N, K, O, BatchCount);
        }
    }
};
