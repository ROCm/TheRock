// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelinePersistentWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelinePersistentWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelinePersistentWmma

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesPersistentWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
