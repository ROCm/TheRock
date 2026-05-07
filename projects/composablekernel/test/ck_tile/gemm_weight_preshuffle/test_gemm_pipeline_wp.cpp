// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineWeightPreshuffle : public TestCkTileGemmPipeline<T>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineWeightPreshuffle

TYPED_TEST_SUITE(TestCkTileGemmPipelineWeightPreshuffle, KernelTypesWeightPreshuffle);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
