// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV4Wmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompV4Wmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV4Wmma

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV4Wmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
