// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineMemWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineMemWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineMemWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineMemWmma, KernelTypesMemWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
