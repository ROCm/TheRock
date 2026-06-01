// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV6
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV6<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV6

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV6);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
