// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV3Wmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompV3Wmma<T>>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base1 = TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompV3Wmma<T>>;
        using Base2 = TestCkTileGemmPipeline<T, Base1>;
        if constexpr(std::is_same_v<typename Base2::BLayout, Row> &&
                     std::is_same_v<typename Base2::BDataType, I4>)
        {
            return false;
        }
        else
        {
            return Base1::check_data_type();
        }
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV3Wmma

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV3Wmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
