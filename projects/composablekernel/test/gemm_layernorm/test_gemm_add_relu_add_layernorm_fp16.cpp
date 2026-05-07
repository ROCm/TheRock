// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_gemm_add_relu_add_layernorm_impl.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestGemmAddReluAddLayernorm : public ::testing::Test
{
    protected:
    using ADataType        = std::tuple_element_t<0, Tuple>;
    using BDataType        = std::tuple_element_t<1, Tuple>;
    using AccDataType      = std::tuple_element_t<2, Tuple>;
    using D0DataType       = std::tuple_element_t<3, Tuple>;
    using D1DataType       = std::tuple_element_t<4, Tuple>;
    using EMeanVarDataType = std::tuple_element_t<5, Tuple>;
    using GammaDataType    = std::tuple_element_t<6, Tuple>;
    using BetaDataType     = std::tuple_element_t<7, Tuple>;
    using HDataType        = std::tuple_element_t<8, Tuple>;
    using ALayout          = std::tuple_element_t<9, Tuple>;
    using BLayout          = std::tuple_element_t<10, Tuple>;
    using D0Layout         = std::tuple_element_t<11, Tuple>;
    using D1Layout         = std::tuple_element_t<12, Tuple>;
    using HLayout          = std::tuple_element_t<13, Tuple>;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {1024, 1024, 1024}, {2048, 640, 640}, {1, 1, 1}};

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = 0;
            int StrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
            int StrideH  = ck::is_same_v<HLayout, Row> ? N : M;

            bool success = ck::profiler::profile_gemm_add_relu_add_layernorm_impl<ADataType,
                                                                                  BDataType,
                                                                                  AccDataType,
                                                                                  D0DataType,
                                                                                  D1DataType,
                                                                                  EMeanVarDataType,
                                                                                  GammaDataType,
                                                                                  BetaDataType,
                                                                                  HDataType,
                                                                                  ALayout,
                                                                                  BLayout,
                                                                                  D0Layout,
                                                                                  D1Layout,
                                                                                  HLayout>(
                true, 1, false, false, M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideH);

            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F32, F16, F16, F16, F16, F16, F16, Row, Row, Row, Row, Row>,
    std::tuple<F16, F16, F32, F16, F16, F16, F16, F16, F16, Row, Col, Row, Row, Row>,
    std::tuple<F16, F16, F32, F16, F16, F16, F16, F16, F16, Col, Row, Row, Row, Row>,
    std::tuple<F16, F16, F32, F16, F16, F16, F16, F16, F16, Col, Col, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmAddReluAddLayernorm, KernelTypes);
TYPED_TEST(TestGemmAddReluAddLayernorm, Test_FP16) { this->Run(); }
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
