// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;
#ifdef CK_ENABLE_FP16
using F16 = ck::half_t;
#endif
using F32 = float;

template <typename Tuple>
class TestSoftmax : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    //         InDataType, AccDataType, OutDataType, Rank
#ifdef CK_ENABLE_FP16
    std::tuple<       F16,         F32,         F16,    I<4>>,
#endif
    std::tuple<       F32,         F32,         F32,    I<4>>
    >;
// clang-format on

TYPED_TEST_SUITE(TestSoftmax, KernelTypes);

#include "test_softmax_ut_cases.inc"
