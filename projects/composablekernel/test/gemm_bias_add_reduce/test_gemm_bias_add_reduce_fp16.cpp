// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_common.hpp"

using F16 = ck::half_t;
using F32 = float;

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
class TestGemmBiasAddReduce_FP16_MK_NK
    : public ck::test::TestGemmBiasAddReduceCommon<
          typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmBiasAddReduce_FP16_MK_KN
    : public ck::test::TestGemmBiasAddReduceCommon<
          typename tuple_concat<std::tuple<Row, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmBiasAddReduce_FP16_KM_KN
    : public ck::test::TestGemmBiasAddReduceCommon<
          typename tuple_concat<std::tuple<Col, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmBiasAddReduce_FP16_KM_NK
    : public ck::test::TestGemmBiasAddReduceCommon<
          typename tuple_concat<std::tuple<Col, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple< F16, F16, F16, F16, F16, F32>
   >;
// clang-format on

TYPED_TEST_SUITE(TestGemmBiasAddReduce_FP16_MK_NK, KernelTypes);
TYPED_TEST_SUITE(TestGemmBiasAddReduce_FP16_MK_KN, KernelTypes);
TYPED_TEST_SUITE(TestGemmBiasAddReduce_FP16_KM_KN, KernelTypes);
TYPED_TEST_SUITE(TestGemmBiasAddReduce_FP16_KM_NK, KernelTypes);

TYPED_TEST(TestGemmBiasAddReduce_FP16_MK_NK, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 1024;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmBiasAddReduce_FP16_MK_KN, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 1024;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmBiasAddReduce_FP16_KM_KN, Regular)
{
    std::vector<int> Ms{256};
    constexpr int N = 512;
    constexpr int K = 1024;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmBiasAddReduce_FP16_KM_NK, Regular)
{
    std::vector<int> Ms{256};
    constexpr int N = 1024;
    constexpr int K = 1024;

    for(int M : Ms)
        this->Run(M, N, K);
}
