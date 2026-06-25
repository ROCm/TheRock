// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using I8   = int8_t;
using I32  = int32_t;
using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using F8   = ck::f8_t;

// M, N, K
using TestMatrixSizes = std::vector<std::vector<ck::index_t>>;

static const TestMatrixSizes DefaultTestMatrixSizes = {
    {16, 32, 64}, {512, 2048, 4096}, {2048, 1024, 16}};

template <typename Tuple>
class TestGemmCommon : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using EDataType   = std::tuple_element_t<3, Tuple>;
    using ALayout     = std::tuple_element_t<4, Tuple>;
    using BLayout     = std::tuple_element_t<5, Tuple>;
    using ELayout     = std::tuple_element_t<6, Tuple>;

    using ProfileCall = bool (*const)(int, int, bool, bool, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run(const TestMatrixSizes& lengths = DefaultTestMatrixSizes)
    {
        bool all_success = true;

        for(auto length : lengths)
        {
            int M       = length[0];
            int N       = length[1];
            int K       = length[2];
            int StrideA = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideE = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success & GetImpl()(1, 1, false, true, M, N, K, StrideA, StrideB, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

template <typename Tuple>
class TestGemmD0Common : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using EDataType   = std::tuple_element_t<4, Tuple>;
    using ALayout     = std::tuple_element_t<5, Tuple>;
    using BLayout     = std::tuple_element_t<6, Tuple>;
    using D0Layout    = std::tuple_element_t<7, Tuple>;
    using ELayout     = std::tuple_element_t<8, Tuple>;

    using ProfileCall = bool (*const)(int, int, bool, bool, int, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run(const TestMatrixSizes& lengths = DefaultTestMatrixSizes)
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
                GetImpl()(1, 1, false, true, M, N, K, StrideA, StrideB, StrideD0, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

template <typename Tuple>
class TestGemmD0D1Common : public ::testing::Test
{
    protected:
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

    using ProfileCall = bool (*const)(int, int, bool, bool, int, int, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run(const TestMatrixSizes& lengths = DefaultTestMatrixSizes)
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
            int StrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                GetImpl()(
                    1, 1, false, true, M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};
