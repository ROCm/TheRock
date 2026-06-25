// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_multiply_multiply_impl.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using I8   = int8_t;
using I32  = int32_t;
using F8   = ck::f8_t;
using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <typename Tuple>
class TestGemmMultiplyMultiply : public ::testing::Test
{
    private:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using D1DataType  = std::tuple_element_t<4, Tuple>;
    using EDataType   = std::tuple_element_t<5, Tuple>;
    using ALayout     = std::tuple_element_t<6, Tuple>;
    using BLayout     = std::tuple_element_t<7, Tuple>;
    using D0Layout    = std::tuple_element_t<8, Tuple>;
    using D1Layout    = std::tuple_element_t<9, Tuple>;
    using ELayout     = std::tuple_element_t<10, Tuple>;

    constexpr static auto ProfileGemmMultiplyMultiplyImpl =
        ck::profiler::profile_gemm_multiply_multiply_impl<ADataType,
                                                          BDataType,
                                                          AccDataType, // ComputeDataType for
                                                                       // reference gemm
                                                          AccDataType,
                                                          D0DataType,
                                                          D1DataType,
                                                          EDataType,
                                                          ALayout,
                                                          BLayout,
                                                          D0Layout,
                                                          D1Layout,
                                                          ELayout>;

    public:
    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {512, 2048, 4096}, {2048, 1024, 16}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
            int StrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success = all_success & ProfileGemmMultiplyMultiplyImpl(1,
                                                                        1,
                                                                        false,
                                                                        true,
                                                                        M,
                                                                        N,
                                                                        K,
                                                                        StrideA,
                                                                        StrideB,
                                                                        StrideD0,
                                                                        StrideD1,
                                                                        StrideE,
                                                                        1,
                                                                        1,
                                                                        1,
                                                                        0);
        }

        EXPECT_TRUE(all_success);
    }
};

using KernelTypes = ::testing::Types<
#ifdef CK_USE_WMMA_FP8
    std::tuple<F8, F8, F32, F32, F32, F16, Row, Col, Row, Col, Row>,
    std::tuple<F8, F8, F32, F32, F32, BF16, Row, Col, Row, Col, Row>,
#endif
    std::tuple<I8, I8, I32, F16, F16, F16, Row, Col, Row, Col, Row>,
    std::tuple<I8, I8, I32, F32, F32, BF16, Row, Col, Row, Col, Row>>;

TYPED_TEST_SUITE(TestGemmMultiplyMultiply, KernelTypes);
TYPED_TEST(TestGemmMultiplyMultiply, Test) { this->Run(); }
