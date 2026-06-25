// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test_batched_gemm_gemm_util.hpp"

template <typename Tuple>
class TestBatchedGemmGemmFP16 : public TestBatchedGemmGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>,
    std::tuple<F16, F16, F16, F16, Row, Col, Col, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmGemmFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16)
{
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 1},
        {128, 128, 136, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->bench_  = true;
    this->verify_ = true;
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 768},
        {256, 256, 128, 128, 768},
        {512, 512, 64, 64, 768},
        {512, 512, 128, 128, 768},
        {1024, 1024, 64, 64, 768},
        {1024, 1024, 128, 128, 768},
        {2048, 2048, 64, 64, 768},
        {2048, 2048, 128, 128, 768},
        {4096, 4096, 64, 64, 768},
        {4096, 4096, 128, 128, 768},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}
