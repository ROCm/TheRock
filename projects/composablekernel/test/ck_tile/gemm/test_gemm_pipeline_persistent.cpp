// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelinePersistent
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelinePersistent<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelinePersistent

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesPersistent);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
