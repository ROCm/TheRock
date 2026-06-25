// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_bilinear_impl.hpp"
#include "test_gemm_common.hpp"

template <typename Tuple>
class TestGemmBilinear : public ::testing::Test
{
    private:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using EDataType   = std::tuple_element_t<4, Tuple>;
    using ALayout     = std::tuple_element_t<5, Tuple>;
    using BLayout     = std::tuple_element_t<6, Tuple>;
    using D0Layout    = std::tuple_element_t<7, Tuple>;
    using ELayout     = std::tuple_element_t<8, Tuple>;

    constexpr static auto ProfileGemmBilinearImpl =
        ck::profiler::profile_gemm_bilinear_impl<ADataType,
                                                 BDataType,
                                                 AccDataType,
                                                 D0DataType,
                                                 EDataType,
                                                 ALayout,
                                                 BLayout,
                                                 D0Layout,
                                                 ELayout>;

    public:
    void Run(TestMatrixSizes const& lengths)
    {
        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                ProfileGemmBilinearImpl(
                    1, 1, false, true, M, N, K, StrideA, StrideB, StrideD0, StrideE, 1.F, 1.F);
        }

        EXPECT_TRUE(all_success);
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16, F32, F16, F16, Row, Row, Row, Row>,
                                     std::tuple<F16, F16, F32, F16, F16, Row, Col, Row, Row>,
                                     std::tuple<F16, F16, F32, F16, F16, Col, Row, Row, Row>,
                                     std::tuple<F16, F16, F32, F16, F16, Col, Col, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, I8, Row, Row, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, I8, Row, Col, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, I8, Col, Row, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, I8, Col, Col, Row, Row>>;

TYPED_TEST_SUITE(TestGemmBilinear, KernelTypes);
TYPED_TEST(TestGemmBilinear, Test) { this->Run(DefaultTestMatrixSizes); }
