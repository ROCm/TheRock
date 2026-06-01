// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsync
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompAsync<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsync

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompAsync);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
