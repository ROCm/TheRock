// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_multiply_multiply_wp_impl.hpp"

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using F32 = float;

template <typename Tuple>
class TestGemmMultiplyMultiplyWPCommon : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using D0Layout        = std::tuple_element_t<2, Tuple>;
    using D1Layout        = std::tuple_element_t<3, Tuple>;
    using ELayout         = Row;
    using ADataType       = std::tuple_element_t<4, Tuple>;
    using BDataType       = std::tuple_element_t<5, Tuple>;
    using ComputeDataType = std::tuple_element_t<6, Tuple>;
    using D0DataType      = std::tuple_element_t<7, Tuple>;
    using D1DataType      = std::tuple_element_t<8, Tuple>;
    using EDataType       = std::tuple_element_t<9, Tuple>;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // decimal value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1, 2, 4}; }

    void Run(const int M, const int N, const int K)
    {
        for(size_t i = 0; i < k_batches_.size(); i++)
        {
            RunSingle(M, N, K, k_batches_[i]);
        }
    }

    void RunSingle(
        const int M, const int N, const int K, int kbatch = 1, int n_warmup = 1, int n_iter = 10)
    {
        bool all_success = true;

        int StrideA  = std::is_same_v<remove_cvref_t<ALayout>, Row> ? K : M;
        int StrideB  = std::is_same_v<remove_cvref_t<BLayout>, Row> ? N : K;
        int StrideD0 = std::is_same_v<remove_cvref_t<D0Layout>, Row> ? N : M;
        int StrideD1 = std::is_same_v<remove_cvref_t<D1Layout>, Row> ? N : M;
        int StrideE  = std::is_same_v<ELayout, Row> ? N : M;

        all_success =
            all_success &
            ck::profiler::profile_gemm_multiply_multiply_weight_preshuffle_impl<ADataType,
                                                                                BDataType,
                                                                                ComputeDataType,
                                                                                F32,
                                                                                D0DataType,
                                                                                D1DataType,
                                                                                EDataType,
                                                                                ALayout,
                                                                                BLayout,
                                                                                D0Layout,
                                                                                D1Layout,
                                                                                ELayout>(
                verify_,
                init_method_,
                log_,
                bench_,
                M,
                N,
                K,
                StrideA,
                StrideB,
                StrideD0,
                StrideD1,
                StrideE,
                kbatch,
                n_warmup,
                n_iter);

        EXPECT_TRUE(all_success);
    }
};

} // namespace test
} // namespace ck
