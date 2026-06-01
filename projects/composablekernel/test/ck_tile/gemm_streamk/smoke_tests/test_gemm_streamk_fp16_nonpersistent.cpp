// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKFp16NonPersistent : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKFp16NonPersistent

TYPED_TEST_SUITE(TestCkTileStreamKFp16NonPersistent, KernelTypesStreamKFp16NonPersistent);

#include "test_gemm_streamk_smoke_cases.inc"

#undef TEST_SUITE_NAME
