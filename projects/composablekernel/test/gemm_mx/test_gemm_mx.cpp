// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "test_gemm_mx_util.hpp"

using E8M0 = ck::e8m0_bexp_t;
using F8   = ck::f8_t;
using BF8  = ck::bf8_t;
using F6   = ck::f6x16_pk_t;
using BF6  = ck::bf6x16_pk_t;
using F4   = ck::f4x2_pk_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

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
class TestGemmMX_MK_NK
    : public ck::test::TestGemmMX<typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmMX_MK_KN
    : public ck::test::TestGemmMX<typename tuple_concat<std::tuple<Row, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmMX_KM_NK
    : public ck::test::TestGemmMX<typename tuple_concat<std::tuple<Col, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    //         ADataType, BDataType,       CDataType, ScaleBlockSize
    std::tuple<       F8,        F8,             F16, ck::Number<32> >,
    std::tuple<       F8,        F8,            BF16, ck::Number<32> >,
#endif
    std::tuple<       F4,        F4,             F16, ck::Number<32> >,
    std::tuple<       F6,        F6,             F16, ck::Number<32> >,
    std::tuple<      BF6,       BF6,            BF16, ck::Number<32> >
    >;

using KernelTypes_MK_KN = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    //         ADataType, BDataType,       CDataType, ScaleBlockSize
    std::tuple<      BF8,        F8,             F16, ck::Number<32> >
#endif
    >;

using KernelTypes_KM_NK = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    //         ADataType, BDataType,       CDataType, ScaleBlockSize
    std::tuple<       F8,        F8,            BF16, ck::Number<32> >
#endif
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmMX_MK_NK, KernelTypes_MK_NK);
TYPED_TEST_SUITE(TestGemmMX_MK_KN, KernelTypes_MK_KN);
TYPED_TEST_SUITE(TestGemmMX_KM_NK, KernelTypes_KM_NK);

/// A: RowMajor
/// B: ColMajor
/// C: RowMajor

TYPED_TEST(TestGemmMX_MK_NK, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 256;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 256;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_NK, Regular)
{
    std::vector<int> Ms{3840};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_NK, Large)
{
    std::vector<std::pair<int, int>> test_sizes{{5120, 5120}, {3840, 5120}, {4096, 4096}};

    constexpr int K       = 4096;
    constexpr int StrideA = K;
    constexpr int StrideB = K;

    for(auto test_size : test_sizes)
    {
        auto M = test_size.first;
        auto N = test_size.second;

        const auto StrideC = N;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

/// A: RowMajor
/// B: RowMajor
/// C: RowMajor

TYPED_TEST(TestGemmMX_MK_KN, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 256;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 256;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_KN, Regular)
{
    std::vector<int> Ms{3840};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_MK_KN, Large)
{
    std::vector<std::pair<int, int>> test_sizes{{5120, 5120}, {3840, 5120}, {4096, 4096}};

    constexpr int K       = 4096;
    constexpr int StrideA = K;

    for(auto test_size : test_sizes)
    {
        auto M = test_size.first;
        auto N = test_size.second;

        const auto StrideB = N;
        const auto StrideC = N;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

/// A: ColMajor
/// B: ColMajor
/// C: RowMajor

TYPED_TEST(TestGemmMX_KM_NK, SmallN)
{
    constexpr int M = 256;
    std::vector<int> Ns{32, 64};
    // std::vector<int> Ns{1, 2, 3, 4, 5, 6};
    constexpr int K = 512;

    constexpr int StrideA = M;
    constexpr int StrideB = K;

    for(int N : Ns)
    {
        const auto StrideC = N;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmMX_KM_NK, MidLargeN)
{
    constexpr int M = 256;
    std::vector<int> Ns{128, 256, 2048};
    // std::vector<int> Ns{127, 255, 312, 799, 1573};
    constexpr int K = 512;

    constexpr int StrideA = M;
    constexpr int StrideB = K;

    for(int N : Ns)
    {
        const auto StrideC = N;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmMX_KM_NK, Regular)
{
    std::vector<int> Ms{3840};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, M, StrideB, StrideC);
}

TYPED_TEST(TestGemmMX_KM_NK, Large)
{
    std::vector<std::pair<int, int>> test_sizes{{5120, 5120}, {3840, 5120}, {4096, 4096}};

    constexpr int K       = 4096;
    constexpr int StrideB = K;

    for(auto test_size : test_sizes)
    {
        auto M = test_size.first;
        auto N = test_size.second;

        const auto StrideA = M;
        const auto StrideC = N;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}
