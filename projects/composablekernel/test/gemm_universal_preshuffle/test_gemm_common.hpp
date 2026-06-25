// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_universal_preshuffle_impl.hpp"

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using F32 = float;

template <typename Tuple>
class TestGemmUniversalPreshuffleCommon : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using CLayout         = Row;
    using ADataType       = std::tuple_element_t<2, Tuple>;
    using BDataType       = std::tuple_element_t<3, Tuple>;
    using ComputeDataType = std::tuple_element_t<4, Tuple>;
    using CDataType       = std::tuple_element_t<5, Tuple>;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1;
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false;
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

        int StrideA = std::is_same_v<ALayout, Row> ? K : M;
        int StrideB = std::is_same_v<BLayout, Row> ? N : K;
        int StrideC = std::is_same_v<CLayout, Row> ? N : M;

        all_success = all_success &
                      ck::profiler::profile_gemm_universal_preshuffle_impl<ADataType,
                                                                           BDataType,
                                                                           ComputeDataType,
                                                                           F32,
                                                                           CDataType,
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
                                                                                    kbatch,
                                                                                    n_warmup,
                                                                                    n_iter);

        EXPECT_TRUE(all_success);
    }
};

} // namespace test
} // namespace ck
