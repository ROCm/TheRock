// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_b_scale_util.hpp"

using I4  = ck::pk_i4_t;
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
class TestGemmBScale_MK_NK
    : public ck::test::TestGemmBScale<typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_NK = ::testing::Types<
    //         ADataType, BDataType, BScaleDataType, ComputeDataType, CDataType
    std::tuple<      F16,        I4,            F16,             F16,       F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmBScale_MK_NK, KernelTypes_MK_NK);

#include "test_gemm_b_scale_ut_cases.inc"
