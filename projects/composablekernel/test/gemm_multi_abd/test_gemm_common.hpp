// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using F32 = float;

template <typename Tuple>
class TestGemmCommon : public ::testing::Test
{
    protected:
    using AsLayout     = std::tuple_element_t<0, Tuple>;
    using BsLayout     = std::tuple_element_t<1, Tuple>;
    using DsLayout     = std::tuple_element_t<2, Tuple>;
    using ELayout      = Row;
    using AsDataType   = std::tuple_element_t<3, Tuple>;
    using BsDataType   = std::tuple_element_t<4, Tuple>;
    using DsDataType   = std::tuple_element_t<5, Tuple>;
    using EDataType    = std::tuple_element_t<6, Tuple>;
    using AElementOp   = std::tuple_element_t<7, Tuple>;
    using BElementOp   = std::tuple_element_t<8, Tuple>;
    using CDEElementOp = std::tuple_element_t<9, Tuple>;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {512, 1024, 2048}, {1024, 512, 32}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M = length[0];
            int N = length[1];
            int K = length[2];
            // Assuming same layout for all A matrices (same applies for Bs and Ds)
            int StrideA = ck::is_same_v<remove_cvref_t<tuple_element_t<0, AsLayout>>, Row> ? K : M;
            int StrideB = ck::is_same_v<remove_cvref_t<tuple_element_t<0, BsLayout>>, Row> ? N : K;
            // In case no D matrices are provided, set stride to 0
            int StrideD = 0;
            if constexpr(DsDataType::Size() > 0)
            {
                StrideD = ck::is_same_v<remove_cvref_t<tuple_element_t<0, DsLayout>>, Row> ? N : M;
            }
            int StrideE = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success & ck::profiler::profile_gemm_multi_abd_impl<AsDataType,
                                                                        BsDataType,
                                                                        F32,
                                                                        DsDataType,
                                                                        EDataType,
                                                                        AsLayout,
                                                                        BsLayout,
                                                                        DsLayout,
                                                                        ELayout,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CDEElementOp>(
                                  1, 2, false, false, M, N, K, StrideA, StrideB, StrideD, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

} // namespace test
} // namespace ck
