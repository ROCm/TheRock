// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using I8  = int8_t;
using I32 = int32_t;

namespace ck {
namespace test {

using TestMatrixSizes = std::vector<std::vector<ck::index_t>>;

static const TestMatrixSizes DefaultTestMatrixSizes = {
    {16, 32, 64}, {512, 2048, 4096}, {2048, 1024, 16}};

template <typename Tuple>
class TestGemmQuantizationCommon : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using EDataType   = std::tuple_element_t<3, Tuple>;
    using ALayout     = std::tuple_element_t<4, Tuple>;
    using BLayout     = std::tuple_element_t<5, Tuple>;
    using ELayout     = std::tuple_element_t<6, Tuple>;

    using ProfileCall = bool (*const)(int, int, bool, bool, int, int, int, int, int, int, float);

    virtual ProfileCall GetImpl() = 0;

    void Run(const TestMatrixSizes& lengths = DefaultTestMatrixSizes)
    {
        bool all_success = true;

        for(auto length : lengths)
        {
            int M               = length[0];
            int N               = length[1];
            int K               = length[2];
            int StrideA         = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB         = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideE         = ck::is_same_v<ELayout, Row> ? N : M;
            float requant_scale = 0.03f;

            all_success =
                all_success &
                GetImpl()(1, 1, false, true, M, N, K, StrideA, StrideB, StrideE, requant_scale);
        }

        EXPECT_TRUE(all_success);
    }
};

} // namespace test
} // namespace ck
