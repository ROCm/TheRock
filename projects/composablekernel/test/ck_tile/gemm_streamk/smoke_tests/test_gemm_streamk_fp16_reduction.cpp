// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp16Reduction : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp16Reduction

TYPED_TEST_SUITE(TestCkTileStreamKFp16Reduction, KernelTypesStreamKFp16Reduction);

#include "test_gemm_streamk_reduction_cases.inc"

#undef TEST_SUITE_NAME
