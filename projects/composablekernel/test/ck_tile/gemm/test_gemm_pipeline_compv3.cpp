// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV3
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV3<T>>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base = TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV3<T>>;
        if constexpr(std::is_same_v<typename Base::ADataType, F8> &&
                     std::is_same_v<typename Base::BDataType, BF8>)
        {
            return false;
        }
        else if constexpr(std::is_same_v<typename Base::BLayout, Row> &&
                          std::is_same_v<typename Base::BDataType, I4>)
        {
            return false;
        }

        return true;
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV3

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV3);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
