// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf16NonPersistent : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf16NonPersistent

TYPED_TEST_SUITE(TestCkTileStreamKBf16NonPersistent, KernelTypesStreamKBf16NonPersistent);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
