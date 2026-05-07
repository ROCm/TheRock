// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_common_includes.hpp"

template <typename Tuple>
class TestCkTileStreamKBf8Persistent : public TestCkTileStreamK<Tuple>
{
};

#define TEST_SUITE_NAME TestCkTileStreamKBf8Persistent

TYPED_TEST_SUITE(TestCkTileStreamKBf8Persistent, KernelTypesStreamKBf8Persistent);

#include "test_gemm_streamk_extended_cases.inc"

#undef TEST_SUITE_NAME
