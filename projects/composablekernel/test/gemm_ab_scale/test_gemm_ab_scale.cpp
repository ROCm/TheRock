// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_ab_scale_util.hpp"

using BF16 = ck::bhalf_t;
using F32  = float;
using F8   = ck::f8_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

} // namespace

template <typename Tuple>
class TestGemmABScale_MK_NK : public ck::test::TestGemmABScale<
                                  typename tuple_concat<std::tuple<Row, Col, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmABScale_MK_KN : public ck::test::TestGemmABScale<
                                  typename tuple_concat<std::tuple<Row, Row, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmABScale_KM_KN : public ck::test::TestGemmABScale<
                                  typename tuple_concat<std::tuple<Col, Row, Row>, Tuple>::type>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, EDataType
    std::tuple<  F8, F32,   F8, F32,              F8,      BF16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmABScale_MK_NK, KernelTypes);
TYPED_TEST_SUITE(TestGemmABScale_MK_KN, KernelTypes);
TYPED_TEST_SUITE(TestGemmABScale_KM_KN, KernelTypes);

// Row Col
TYPED_TEST(TestGemmABScale_MK_NK, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_NK, SmallMPadK)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 704;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_NK, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideE = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideE);
}

// Row Row
TYPED_TEST(TestGemmABScale_MK_KN, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_KN, SmallMPadK)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 704;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmABScale_MK_KN, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideE = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideE);
}

// Col Row
TYPED_TEST(TestGemmABScale_KM_KN, SmallM)
{
    std::vector<int> Ms{16, 32};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmABScale_KM_KN, SmallMPadK)
{
    std::vector<int> Ms{16, 32};
    constexpr int N = 512;
    constexpr int K = 704;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmABScale_KM_KN, MidLargeM)
{
    std::vector<int> Ms{128, 256};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmABScale_KM_KN, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideB = N;
    constexpr int StrideE = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideE);
    }
}
