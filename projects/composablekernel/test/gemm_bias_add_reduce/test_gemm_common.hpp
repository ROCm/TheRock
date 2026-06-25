// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_bias_add_reduce_impl.hpp"

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using F32 = float;

template <typename Tuple>
class TestGemmBiasAddReduceCommon : public ::testing::Test
{
    protected:
    using ALayout        = std::tuple_element_t<0, Tuple>;
    using BLayout        = std::tuple_element_t<1, Tuple>;
    using CLayout        = Row;
    using ADataType      = std::tuple_element_t<2, Tuple>;
    using BDataType      = std::tuple_element_t<3, Tuple>;
    using CDataType      = std::tuple_element_t<4, Tuple>;
    using BiasDataType   = std::tuple_element_t<5, Tuple>;
    using D0DataType     = std::tuple_element_t<6, Tuple>;
    using ReduceDataType = std::tuple_element_t<7, Tuple>;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance

    void Run(const int M, const int N, const int K)
    {
        bool all_success = true;

        int StrideA  = std::is_same_v<remove_cvref_t<ALayout>, Row> ? K : M;
        int StrideB  = std::is_same_v<remove_cvref_t<BLayout>, Row> ? N : K;
        int StrideD0 = std::is_same_v<remove_cvref_t<CLayout>, Row> ? N : M;
        int StrideC  = std::is_same_v<CLayout, Row> ? N : M;

        all_success =
            all_success &
            ck::profiler::profile_gemm_bias_add_reduce_impl<ADataType,
                                                            BDataType,
                                                            CDataType,
                                                            BiasDataType,
                                                            D0DataType,
                                                            ReduceDataType,
                                                            ALayout,
                                                            BLayout,
                                                            CLayout>(
                verify_, init_method_, log_, bench_, M, N, K, StrideA, StrideB, StrideC, StrideD0);

        EXPECT_TRUE(all_success);
    }
};

} // namespace test
} // namespace ck
