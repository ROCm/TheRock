// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_common.hpp"

using F8   = ck::f8_t;
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
class TestGemmBlockScaleWP_FP8_MK_NK : public ck::test::TestGemmBlockscaleWPCommon<
                                           typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
#if defined(CK_ENABLE_FP8)
    std::tuple< F8, F32, F8, F32, F8, BF16>
#endif
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmBlockScaleWP_FP8_MK_NK, KernelTypes_MK_NK);

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, Regular0)
{
    std::vector<int> Ms{128, 256, 512};
    constexpr int N = 512;
    constexpr int K = 2048;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TestGemmBlockScaleWP_FP8_MK_NK, Regular1)
{
    std::vector<int> Ms{128, 256, 512};
    constexpr int N = 1024;
    constexpr int K = 4096;

    for(int M : Ms)
        this->Run(M, N, K);
}
